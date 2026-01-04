import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from api.workflows.tree_of_thoughts_research import TreeOfThoughtsResearch
from open_notebook.domain.notebook import Note, Notebook

router = APIRouter()

# Pydantic models for request/response
class DeepResearchRequest(BaseModel):
    query: str
    notebook_id: Optional[str] = None
    max_iterations: int = 3
    save_to_notebook: bool = True
    model_id: str = None  # Model ID from database

class DeepResearchResponse(BaseModel):
    research_id: str
    status: str
    synthesis: str
    confidence_score: float
    sources_count: int
    iterations: int

class DeepResearchProgressResponse(BaseModel):
    research_id: str
    status: str
    iteration: int
    active_branches: int
    message: str

# Store for in-progress research (in production, use Redis)
research_progress = {}
research_results = {}

def get_workflow(model_id: str = None):
    """Dependency to get the workflow instance"""
    from open_notebook.domain.models import model_manager
    import asyncio
    
    async def initialize_workflow():
        try:
            # If model_id is provided, get the model from database
            # Otherwise, get the default chat model
            if model_id:
                llm = await model_manager.get_model(model_id)
            else:
                defaults = await model_manager.get_defaults()
                if defaults.default_chat_model:
                    llm = await model_manager.get_model(defaults.default_chat_model)
                else:
                    # Fallback to first available language model
                    from open_notebook.domain.models import Model
                    models = await Model.get_models_by_type("language")
                    if not models:
                        raise ValueError("No language models configured")
                    llm = await model_manager.get_model(models[0].id)
            
            # Get embedding model for vector search
            embedding_model = await model_manager.get_embedding_model()
            if not embedding_model:
                raise ValueError("No embedding model configured")
            
            # Convert Esperanto model to LangChain model for compatibility
            langchain_llm = llm.to_langchain()
            
            return TreeOfThoughtsResearch(llm=langchain_llm, embeddings=embedding_model)
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {e}")
            raise
    
    # Run async initialization synchronously for FastAPI dependency
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(initialize_workflow())

@router.get("/deep-research/available-models")
async def get_available_models():
    """Get all available language models for deep research"""
    try:
        from open_notebook.domain.models import Model, model_manager
        
        # Get all language models
        models = await Model.get_models_by_type("language")
        
        # Get the default model ID
        default_model_id = None
        try:
            defaults = await model_manager.get_defaults()
            if defaults.default_chat_model:
                default_model_id = defaults.default_chat_model
            elif models:
                default_model_id = models[0].id
        except Exception:
            if models:
                default_model_id = models[0].id
        
        return {
            'models': [
                {
                    'id': model.id,
                    'name': model.name,
                    'provider': model.provider
                }
                for model in models
            ],
            'default_model_id': default_model_id
        }
    except Exception as e:
        logger.error(f"Error fetching available models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")

@router.post("/deep-research", response_model=dict)
async def start_deep_research(
    request: DeepResearchRequest,
    background_tasks: BackgroundTasks,
    workflow: TreeOfThoughtsResearch = Depends(get_workflow)
):
    """Start a new deep research operation"""
    research_id = str(uuid.uuid4())
    
    logger.info(f"Starting deep research {research_id} for query: {request.query}")
    
    # Initialize progress tracking
    research_progress[research_id] = {
        'status': 'started',
        'iteration': 0,
        'active_branches': 0,
        'message': 'Initializing research...'
    }
    
    async def run_research():
        try:
            research_progress[research_id]['status'] = 'processing'
            research_progress[research_id]['message'] = 'Generating hypotheses...'
            
            result = await workflow.run(
                query=request.query,
                research_id=research_id,
                max_depth=request.max_iterations,
                model_id=request.model_id
            )
            
            logger.info(f"Research {research_id} completed successfully")
            
            # Store result
            research_results[research_id] = {
                'query': request.query,
                'synthesis': result['synthesis'],
                'confidence_score': result['confidence_score'],
                'sources_count': result['sources_count'],
                'iterations': result['iterations'],
                'thought_tree': result['thought_tree'],
                'best_path': result['best_path'],
                'created_at': datetime.utcnow().isoformat(),
                'notebook_id': request.notebook_id
            }
            
            # Update progress
            research_progress[research_id] = {
                'status': 'completed',
                'iteration': result['iterations'],
                'message': 'Research completed',
                'synthesis': result['synthesis'],
                'confidence_score': result['confidence_score']
            }
            
            # Save to notebook if requested
            if request.save_to_notebook and request.notebook_id:
                await save_research_to_notebook(
                    notebook_id=request.notebook_id,
                    research_id=research_id,
                    result=result
                )
                
        except Exception as e:
            logger.error(f"Research {research_id} failed: {str(e)}")
            
            research_progress[research_id] = {
                'status': 'failed',
                'error': str(e),
                'message': f'Research failed: {str(e)}'
            }
    
    background_tasks.add_task(run_research)
    
    return {
        'research_id': research_id,
        'status': 'started',
        'message': 'Deep research initiated. Use the research_id to track progress.'
    }

@router.get("/deep-research/{research_id}")
async def get_research_result(research_id: str):
    """Get completed research results"""
    if research_id not in research_results:
        # Check if still processing
        if research_id in research_progress:
            progress = research_progress[research_id]
            if progress['status'] != 'completed':
                raise HTTPException(
                    status_code=202,
                    detail=f"Research status: {progress['status']}"
                )
        else:
            raise HTTPException(status_code=404, detail="Research not found")
    
    result = research_results[research_id]
    return DeepResearchResponse(
        research_id=research_id,
        status='completed',
        synthesis=result['synthesis'],
        confidence_score=result['confidence_score'],
        sources_count=result['sources_count'],
        iterations=result['iterations']
    )

@router.get("/deep-research/{research_id}/progress")
async def get_research_progress(research_id: str):
    """Get research progress without blocking"""
    if research_id not in research_progress:
        raise HTTPException(status_code=404, detail="Research not found")
    
    progress = research_progress[research_id]
    return DeepResearchProgressResponse(
        research_id=research_id,
        status=progress['status'],
        iteration=progress.get('iteration', 0),
        active_branches=progress.get('active_branches', 0),
        message=progress.get('message', '')
    )

@router.get("/deep-research/{research_id}/stream")
async def stream_research_progress(research_id: str):
    """Stream research progress updates via Server-Sent Events"""
    async def event_generator():
        import asyncio
        while True:
            if research_id in research_progress:
                progress = research_progress[research_id]
                yield f"data: {json.dumps(progress)}\n\n"
                
                if progress['status'] in ['completed', 'failed']:
                    break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/deep-research/{research_id}/full")
async def get_full_research_details(research_id: str):
    """Get full research details including thought tree"""
    if research_id not in research_results:
        raise HTTPException(status_code=404, detail="Research not found")
    
    result = research_results[research_id]
    return {
        'id': research_id,
        'query': result['query'],
        'status': 'completed',
        'synthesis': result['synthesis'],
        'confidence_score': result['confidence_score'],
        'sources_count': result['sources_count'],
        'iterations': result['iterations'],
        'thought_tree': result['thought_tree'],
        'best_path': result['best_path'],
        'created_at': result['created_at']
    }

@router.delete("/deep-research/{research_id}")
async def delete_research(research_id: str):
    """Delete a research record"""
    if research_id not in research_progress and research_id not in research_results:
        raise HTTPException(status_code=404, detail="Research not found")
    
    # Clean up progress tracking and results
    if research_id in research_progress:
        del research_progress[research_id]
    if research_id in research_results:
        del research_results[research_id]
    
    return {'message': 'Research deleted successfully'}

async def save_research_to_notebook(
    notebook_id: str,
    research_id: str,
    result: dict
):
    """Save research results as a new note in the notebook"""
    try:
        note_content = f"""# Deep Research Results

## Query
{result['query']}

## Summary
{result['synthesis']}

## Confidence Score
{result['confidence_score']:.1%}

## Research Statistics
- Total Sources Analyzed: {result['sources_count']}
- Research Iterations: {result['iterations']}
- Hypotheses Explored: {len(result.get('thought_tree', []))}

## Methodology
This research was conducted using Tree-of-Thoughts (ToT) methodology with ReAct loops, exploring multiple hypotheses in parallel and synthesizing findings from the most supported branches.

---
*Generated by Deep Research on {datetime.utcnow().isoformat()}*
*Research ID: {research_id}*
"""
        
        note = Note(
            title=f"Deep Research: {result['query'][:50]}...",
            content=note_content,
            note_type="ai",
            research_id=research_id
        )
        
        await note.save()
        await note.add_to_notebook(notebook_id)
        
        logger.info(f"Research {research_id} saved to notebook {notebook_id}")
        
    except Exception as e:
        logger.error(f"Failed to save research to notebook: {e}")
