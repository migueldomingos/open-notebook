"""
Test-Time Diffusion Deep Researcher (TTD-DR)
Based on: https://arxiv.org/abs/2507.16075

Algorithm mimics human research process:
1. Create initial research plan (key areas to cover)
2. Generate rough draft based on plan
3. Iteratively refine draft through:
   - Component-wise self-evolution (multiple variants with feedback)
   - Report-level denoising via retrieval (search → refine → repeat)
4. Generate final polished report

Key insight: Treat research as diffusion - noisy draft → gradually refined → polished report
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
import json
import asyncio
from datetime import datetime
import logging
from open_notebook.database.repository import repo_query

logger = logging.getLogger(__name__)

class SearchQuestion(TypedDict):
    question: str
    context: str
    iteration: int

class AnswerVariant(TypedDict):
    content: str
    sources: List[str]
    fitness_score: float
    feedback: str

class SearchAnswer(TypedDict):
    question: str
    answer: str
    sources: List[str]
    iteration: int

class ResearchPlan(TypedDict):
    key_areas: List[str]
    outline: str

class TTDDRState(TypedDict):
    query: str
    research_plan: ResearchPlan
    draft: str
    draft_history: List[str]
    search_questions: List[SearchQuestion]
    search_answers: List[SearchAnswer]
    current_iteration: int
    max_iterations: int
    final_report: str
    confidence_score: float
    research_id: str
    status: str
    sources_analyzed: int
    all_variants_history: List[Dict[str, Any]]

class TTDDeepResearch:
    """Test-Time Diffusion Deep Researcher"""
    
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.graph = self._build_graph()
        self.progress_callback = None  # Optional callback for progress updates
    
    def set_progress_callback(self, callback):
        """Set a callback function to receive progress updates
        
        Callback signature: callback(research_id: str, stage: str, message: str, **kwargs)
        """
        self.progress_callback = callback
    
    def _update_progress(self, research_id: str, stage: str, message: str, **kwargs):
        """Internal helper to update progress"""
        if self.progress_callback:
            self.progress_callback(research_id=research_id, stage=stage, message=message, **kwargs)
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(TTDDRState)
        
        # Stage 1: Research Plan Generation
        workflow.add_node("generate_research_plan", self.generate_research_plan)
        
        # Stage 2a: Iterative Search Question Generation
        workflow.add_node("generate_search_questions", self.generate_search_questions)
        
        # Stage 2b: Answer Searching with Self-Evolution
        workflow.add_node("search_and_evolve_answers", self.search_and_evolve_answers)
        
        # Report-level Denoising with Retrieval
        workflow.add_node("refine_draft", self.refine_draft)
        
        # Iteration control
        workflow.add_node("check_iteration", self.check_iteration)
        
        # Stage 3: Final Report Generation
        workflow.add_node("generate_final_report", self.generate_final_report)
        
        # Edges
        workflow.add_edge("generate_research_plan", "generate_search_questions")
        workflow.add_edge("generate_search_questions", "search_and_evolve_answers")
        workflow.add_edge("search_and_evolve_answers", "refine_draft")
        workflow.add_edge("refine_draft", "check_iteration")
        
        workflow.add_conditional_edges(
            "check_iteration",
            lambda state: "continue" if state["current_iteration"] < state["max_iterations"] else "synthesize",
            {
                "continue": "generate_search_questions",
                "synthesize": "generate_final_report"
            }
        )
        
        workflow.add_edge("generate_final_report", END)
        
        workflow.set_entry_point("generate_research_plan")
        return workflow.compile()
    
    async def generate_research_plan(self, state: TTDDRState) -> TTDDRState:
        """Stage 1: Generate structured research plan with key areas"""
        logger.info(f"🔬 Stage 1: Generating research plan for: {state['query']}")
        self._update_progress(state['research_id'], 'plan_generation', 'Analyzing query and generating research plan...')
        
        prompt = f"""Você é um especialista em planeamento de pesquisa. Crie um plano de pesquisa detalhado para esta consulta:
        
Consulta: "{state['query']}"

Gere um plano de pesquisa abrangente com:
1. Áreas-chave a investigar (5-7 tópicos principais)
2. Um esboço estruturado de como abordar a pesquisa
3. Subperguntas importantes a responder

Retorne como JSON:
{{
  "key_areas": ["área1", "área2", "área3", ...],
  "outline": "Esboço detalhado da abordagem da pesquisa...",
  "sub_questions": ["pergunta1", "pergunta2", ...]
}}"""
        
        try:
            logger.debug("🔬 Generating research plan from query")
            self._update_progress(state['research_id'], 'plan_generation', 'Calling LLM to generate research plan...')
            
            response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=prompt)]), timeout=60.0)
            response_text = response.content.strip() if response.content else ''
            
            logger.info(f"🔬 LLM response received: {repr(response_text[:200] if response_text else 'EMPTY')}")
            self._update_progress(state['research_id'], 'plan_generation', 'Parsing research plan JSON...')
            
            if not response_text:
                raise ValueError("Empty response from LLM")
            
            # Remove thinking/reasoning tags before JSON parsing
            response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
            
            # Extract JSON from code blocks (proven pattern from tree_of_thoughts_research.py)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Direct JSON parsing with proper error handling
            plan_data = json.loads(response_text)
            
            state['research_plan'] = {
                'key_areas': plan_data.get('key_areas', [state['query']]),
                'outline': plan_data.get('outline', f"Research about {state['query']}")
            }
            
            # Generate initial rough draft from plan
            draft_prompt = f"""Com base neste plano de pesquisa para "{state['query']}"

Áreas-chave: {', '.join(state['research_plan']['key_areas'])}
Esboço: {state['research_plan']['outline']}

Escreva um esboço inicial bruto que delineia o que a pesquisa abordará. Esta é uma versão inicial "ruidosa" que será refinada.
Mantenha-a estruturada com secções claras para cada área-chave."""
            
            logger.debug("📝 Generating initial draft")
            draft_response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=draft_prompt)]), timeout=60.0)
            draft_text = draft_response.content.strip() if draft_response.content else ''
            # Remove thinking tags from draft
            draft_text = draft_text.replace('<think>', '').replace('</think>', '').strip()
            logger.info(f"📝 Draft response received: {repr(draft_text[:200] if draft_text else 'EMPTY')}")
            state['draft'] = draft_text
            state['draft_history'] = [state['draft']]
            
            logger.info(f"✅ Research plan generated with {len(state['research_plan']['key_areas'])} key areas")
            
        except asyncio.TimeoutError:
            logger.error("❌ Research plan generation timeout")
            state['research_plan'] = {
                'key_areas': [state['query']],
                'outline': f"Research about {state['query']}"
            }
            state['draft'] = f"Researching: {state['query']}"
            state['draft_history'] = [state['draft']]
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"❌ Failed to generate research plan: {e}")
            logger.error(f"🔬 Response text that failed JSON parse: {repr(response_text[:500] if 'response_text' in locals() else 'response_text not defined')}")
            # Fallback: create a basic research plan from the query itself
            logger.warning(f"⚠️  Using fallback research plan for query: {state['query']}")
            state['research_plan'] = {
                'key_areas': [
                    state['query'],
                    f"Advanced topics in {state['query']}",
                    f"Practical applications of {state['query']}"
                ],
                'outline': f"Comprehensive research about: {state['query']}"
            }
            state['draft'] = f"Initial draft about {state['query']}"
            state['draft_history'] = [state['draft']]
        
        return state
    
    async def generate_search_questions(self, state: TTDDRState) -> TTDDRState:
        """Stage 2a: Generate search questions based on draft and plan"""
        logger.info(f"👵 Stage 2a (Iteration {state['current_iteration']}): Generating search questions")
        
        # Build context from draft and previous answers
        context_parts = [
            f"Query: {state['query']}",
            f"Current draft:\n{state['draft']}"
        ]
        
        if state['search_answers']:
            context_parts.append(f"Previous answers found:\n" + "\n".join(
                f"Q: {ans['question']}\nA: {ans['answer'][:200]}..."
                for ans in state['search_answers'][-3:]  # Last 3 answers
            ))
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Você é um assistente de pesquisa. Com base no esboço actual e no plano de pesquisa, 
quais são as perguntas mais importantes que precisam ser respondidas a seguir?

{context}

Gere 2-3 perguntas de pesquisa que ajudariam a:
1. Preencher lacunas no esboço actual
2. Verificar afirmações existentes
3. Aprofundar a compreensão das áreas-chave

Retorne como matriz JSON:
[
  {{"question": "pergunta específica?", "context": "por que esta pergunta é importante"}},
  ...
]"""
        
        try:
            logger.debug("👵 Generating search questions")
            self._update_progress(state['research_id'], 'search_questions', f'Generating search questions for iteration {state["current_iteration"]+1}...', iteration=state["current_iteration"]+1)
            
            response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=prompt)]), timeout=60.0)
            response_text = response.content.strip() if response.content else ''
            
            if not response_text:
                raise ValueError("Empty response from LLM")
            
            # Remove thinking/reasoning tags before JSON parsing
            response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
            
            # Extract JSON from code blocks (proven pattern)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Direct JSON parsing with proper error handling
            questions_data = json.loads(response_text)
            
            # Ensure it's a list
            if not isinstance(questions_data, list):
                questions_data = [questions_data] if isinstance(questions_data, dict) else []
            
            state['search_questions'] = [
                {
                    'question': q.get('question', '') if isinstance(q, dict) else str(q),
                    'context': q.get('context', '') if isinstance(q, dict) else '',
                    'iteration': state['current_iteration']
                }
                for q in questions_data if q
            ]
            
            logger.info(f"✅ Generated {len(state['search_questions'])} search questions")
            
        except asyncio.TimeoutError:
            logger.error("❌ Search questions generation timeout")
            state['search_questions'] = [{
                'question': state['query'],
                'context': 'Fallback question',
                'iteration': state['current_iteration']
            }]
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"❌ Failed to generate search questions: {e}")
            state['search_questions'] = [
                {
                    'question': f"What is known about {state['query']}?",
                    'context': "Initial exploration",
                    'iteration': state['current_iteration']
                }
            ]
        
        return state
    
    async def search_and_evolve_answers(self, state: TTDDRState) -> TTDDRState:
        """Stage 2b: Search for answers and apply component-wise self-evolution (PARALLEL)"""
        logger.info(f"🔍 Stage 2b (Iteration {state['current_iteration']}): Searching and evolving answers (parallel mode)")
        self._update_progress(state['research_id'], 'knowledge_search', f'Starting parallel knowledge base search for {len(state["search_questions"])} questions...', iteration=state["current_iteration"]+1)
        
        all_iteration_answers = []
        
        # Process all questions in parallel
        async def process_single_question(idx: int, question_obj: dict) -> tuple:
            """Process a single question: search, generate variants, evolve, merge"""
            question = question_obj['question']
            logger.info(f"📌 [Parallel Q{idx+1}] Processing: {question}")
            
            try:
                # Parallel document search + document retrieval (these are independent)
                self._update_progress(state['research_id'], 'knowledge_search', f'[Parallel] Q{idx+1}: Searching documents...')
                documents = await self._search_knowledge_base(question)
                logger.info(f"📄 [Q{idx+1}] Found {len(documents)} documents")
                
                # Update sources count (thread-safe since just incrementing)
                state['sources_analyzed'] += len(documents)
                self._update_progress(state['research_id'], 'knowledge_search', f'[Parallel] Q{idx+1}: Found {len(documents)} docs, generating variants...', search_progress={'found_results': len(documents)})
                
                # Generate variants
                logger.debug(f"🎨 [Q{idx+1}] Generating answer variants")
                answer_variants = await self._generate_answer_variants(question, documents, num_variants=3)
                logger.info(f"✅ [Q{idx+1}] Generated {len(answer_variants)} variants")
                self._update_progress(state['research_id'], 'knowledge_search', f'[Parallel] Q{idx+1}: Generated variants, now evolving...')
                
                # Self-evolution loop
                logger.debug(f"🔄 [Q{idx+1}] Evolving variants")
                evolved_variants = await self._evolve_answer_variants(
                    question, 
                    answer_variants,
                    documents,
                    max_evolution_iterations=2
                )
                logger.info(f"✅ [Q{idx+1}] Evolved to {len(evolved_variants)} variants")
                self._update_progress(state['research_id'], 'knowledge_search', f'[Parallel] Q{idx+1}: Evolved variants, merging...')
                
                # Merge variants
                logger.debug(f"🔀 [Q{idx+1}] Merging variants")
                final_answer = await self._merge_variants(evolved_variants, question)
                logger.info(f"✅ [Q{idx+1}] Merged answer")
                
                return (idx, question, final_answer)
                
            except Exception as e:
                logger.error(f"❌ [Q{idx+1}] Error processing question: {e}")
                return (idx, question, {'answer': f'Error: {str(e)}', 'sources': []})
        
        # Launch all questions in parallel
        logger.info(f"🚀 Launching {len(state['search_questions'])} parallel question processors...")
        tasks = [
            process_single_question(idx, question_obj)
            for idx, question_obj in enumerate(state['search_questions'])
        ]
        
        results = await asyncio.gather(*tasks)
        logger.info(f"✅ All {len(results)} questions processed in parallel")
        
        # Collect results in order
        for idx, question, final_answer in sorted(results, key=lambda x: x[0]):
            search_answer = SearchAnswer(
                question=question,
                answer=final_answer['answer'],
                sources=final_answer['sources'],
                iteration=state['current_iteration']
            )
            state['search_answers'].append(search_answer)
            all_iteration_answers.append(final_answer)
        
        # Store variant history for analysis
        state['all_variants_history'].append({
            'iteration': state['current_iteration'],
            'answers': all_iteration_answers
        })
        
        logger.info(f"✅ Completed parallel answer searching and self-evolution for iteration {state['current_iteration']}")
        return state
    
    async def _search_knowledge_base(self, query: str, limit: int = 5) -> List[Document]:
        """Search for relevant documents"""
        try:
            # Debug: Log available methods
            logger.debug(f"🧬 Embedding model type: {type(self.embeddings)}")
            logger.debug(f"🧬 Available embedding methods: {[m for m in dir(self.embeddings) if 'embed' in m.lower()]}")
            
            query_embedding = None
            
            # Handle both async and sync embedding methods
            if hasattr(self.embeddings, 'aembed'):
                # Async batch embed - returns list of embeddings
                result = await self.embeddings.aembed([query])
                if result and len(result) > 0:
                    query_embedding = result[0]
            elif hasattr(self.embeddings, 'aembed_query'):
                # Async single query embed
                query_embedding = await self.embeddings.aembed_query(query)
            elif hasattr(self.embeddings, 'embed_query'):
                # Fallback to sync method wrapped in async
                query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
            elif hasattr(self.embeddings, 'embed'):
                # Fallback to sync batch embed
                result = await asyncio.to_thread(self.embeddings.embed, [query])
                if result and len(result) > 0:
                    query_embedding = result[0]
            else:
                raise ValueError(f"Embedding model has no compatible embedding method. Available: {[m for m in dir(self.embeddings) if 'embed' in m.lower()]}")
            
            # Validate embedding was generated
            if not query_embedding:
                logger.warning(f"⚠️  No embedding generated for query: {query}. Returning empty documents.")
                return []
            
            # Use SurrealDB's built-in vector_search function (proven to work)
            search_results = await repo_query(
                """
                SELECT * FROM fn::vector_search($embed, $results, true, false, $minimum_score);
                """,
                {
                    "embed": query_embedding,
                    "results": limit,
                    "minimum_score": 0.3,
                },
            )
            
            documents = []
            if search_results and isinstance(search_results, list):
                for result in search_results:
                    if not isinstance(result, dict):
                        continue
                    
                    doc = Document(
                        page_content=result.get('full_text', result.get('content', ''))[:500],
                        metadata={
                            'source': str(result.get('id', '')),
                            'title': result.get('title', ''),
                            'relevance': result.get('score', 0)
                        }
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []
    
    async def _generate_answer_variants(self, question: str, documents: List[Document], num_variants: int = 3) -> List[AnswerVariant]:
        """Generate multiple initial answer variants"""
        variants = []
        
        for i in range(num_variants):
            # Each variant explores different aspects
            perspective = ["abrangente", "técnica", "prática"][i % 3]
            
            prompt = f"""Responda a esta pergunta de pesquisa de uma perspectiva {perspective}:

Pergunta: {question}

Informações disponíveis:
{json.dumps([
    {
        'title': doc.metadata.get('title', ''),
        'content': doc.page_content[:300]
    }
    for doc in documents[:5]
], ensure_ascii=False)}

Forneça uma resposta detalhada que sintetize as informações disponíveis."""
            
            try:
                logger.debug(f"🎨 Generating variant {i+1}/{num_variants} for question")
                response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=prompt)]), timeout=60.0)
                content = response.content.strip()
                # Remove thinking tags
                content = content.replace('<think>', '').replace('</think>', '').strip()
                variant = AnswerVariant(
                    content=content,
                    sources=[doc.metadata.get('source', '') for doc in documents],
                    fitness_score=0.5,  # Will be updated after evaluation
                    feedback=""
                )
                variants.append(variant)
                logger.debug(f"✅ Variant {i+1} generated successfully")
            except asyncio.TimeoutError:
                logger.error(f"❌ LLM call timeout generating variant {i}")
                variants.append(AnswerVariant(content="Unable to generate (timeout)", sources=[], fitness_score=0.0, feedback=""))
            except Exception as e:
                logger.warning(f"⚠️  Failed to generate variant {i}: {e}")
        
        return variants
    
    async def _evolve_answer_variants(self, question: str, variants: List[AnswerVariant], documents: List[Document], max_evolution_iterations: int = 2) -> List[AnswerVariant]:
        """Apply self-evolution: evaluate, get feedback, revise, repeat"""
        evolved_variants = variants.copy()
        
        for iteration in range(max_evolution_iterations):
            logger.debug(f"🔄 Evolution iteration {iteration + 1}/{max_evolution_iterations}")
            
            # Environmental feedback: Judge each variant
            for variant in evolved_variants:
                feedback_prompt = f"""Avalie esta resposta à pergunta de pesquisa:

Pergunta: {question}
Resposta: {variant['content'][:500]}

Avalie:
1. Utilidade (0-1): Quão bem responde à pergunta?
2. Abrangência (0-1): Cobre os aspectos-chave?
3. Precisão (0-1): A informação é provavelmente precisa?

Retorne JSON:
{{
  "helpfulness": 0.8,
  "comprehensiveness": 0.7,
  "accuracy": 0.85,
  "feedback": "Comentários específicos para melhoria..."
}}"""
                
                try:
                    logger.debug(f"💬 Getting feedback for variant")
                    response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=feedback_prompt)]), timeout=60.0)
                    response_text = response.content.strip() if response.content else ''
                    
                    if not response_text:
                        raise ValueError("Empty response from LLM")
                    
                    # Remove thinking/reasoning tags before JSON parsing
                    response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
                    
                    # Extract JSON from code blocks
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                    
                    # Try to parse JSON with multiple fallbacks
                    feedback_data = None
                    import re
                    import json as json_module
                    
                    # Attempt 1: Direct JSON parse
                    try:
                        feedback_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        # Attempt 2: Extract from code blocks
                        for delimiter in ['```json', '```']:
                            if delimiter in response_text:
                                try:
                                    extracted = response_text.split(delimiter)[1].split('```')[0].strip()
                                    feedback_data = json.loads(extracted)
                                    break
                                except (json.JSONDecodeError, IndexError):
                                    continue
                        
                        # Attempt 3: Find JSON object using regex (greedy to catch incomplete JSON)
                        if not feedback_data:
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    feedback_data = json.loads(json_match.group())
                                except json.JSONDecodeError:
                                    # Attempt 4: Clean up common JSON issues
                                    try:
                                        dirty_json = json_match.group()
                                        # Remove trailing commas before closing braces
                                        dirty_json = re.sub(r',\s*([}\]])', r'\1', dirty_json)
                                        # Remove comments
                                        dirty_json = re.sub(r'//.*?\n', '\n', dirty_json)
                                        feedback_data = json.loads(dirty_json)
                                    except json.JSONDecodeError:
                                        pass
                    
                    if not feedback_data or not isinstance(feedback_data, dict):
                        # Provide default values if parsing fails completely
                        logger.debug(f"⚠️  Failed to parse feedback JSON, using defaults. Raw response: {response_text[:200]}")
                        feedback_data = {
                            'helpfulness': 0.5,
                            'comprehensiveness': 0.5,
                            'accuracy': 0.5,
                            'feedback': 'Could not parse LLM feedback'
                        }
                    
                    # Calculate fitness score with safe defaults
                    variant['fitness_score'] = (
                        feedback_data.get('helpfulness', 0.5) * 0.4 +
                        feedback_data.get('comprehensiveness', 0.5) * 0.4 +
                        feedback_data.get('accuracy', 0.5) * 0.2
                    )
                    variant['feedback'] = feedback_data.get('feedback', '')
                    logger.debug(f"✅ Feedback generated successfully, fitness_score={variant['fitness_score']:.2f}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"❌ Feedback generation timeout")
                    variant['fitness_score'] = 0.5
                except Exception as e:
                    logger.warning(f"⚠️  Feedback generation failed: {e}")
                    variant['fitness_score'] = 0.5
            
            # Revision: Improve variants based on feedback
            if iteration < max_evolution_iterations - 1:  # Don't revise in last iteration
                for i, variant in enumerate(evolved_variants):
                    if variant['fitness_score'] < 0.85:  # Room for improvement
                        revision_prompt = f"""Revise esta resposta para melhorar com base nos comentários:

Pergunta: {question}
Resposta actual: {variant['content'][:400]}
Comentários: {variant['feedback']}

Forneça uma resposta melhorada que aborde os comentários."""
                        
                        try:
                            logger.debug(f"✏️  Generating revision for variant")
                            response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=revision_prompt)]), timeout=60.0)
                            content = response.content.strip()
                            # Remove thinking tags
                            content = content.replace('<think>', '').replace('</think>', '').strip()
                            variant['content'] = content
                            logger.debug(f"✅ Revision generated successfully")
                        except asyncio.TimeoutError:
                            logger.error(f"❌ Revision generation timeout, keeping original content")
                        except Exception as e:
                            logger.warning(f"⚠️  Revision failed: {e}")
        
        return evolved_variants
    
    async def _merge_variants(self, variants: List[AnswerVariant], question: str) -> Dict[str, Any]:
        """Cross-over: Merge multiple evolved variants into single high-quality answer"""
        if not variants:
            return {'answer': '', 'sources': []}
        
        if len(variants) == 1:
            return {
                'answer': variants[0]['content'],
                'sources': variants[0]['sources']
            }
        
        # Sort by fitness score
        sorted_variants = sorted(variants, key=lambda v: v['fitness_score'], reverse=True)
        
        merge_prompt = f"""Fundir estas respostas de especialistas numa única resposta abrangente:

Pergunta: {question}

{json.dumps([
    {
        'answer': v['content'][:300],
        'quality': v['fitness_score']
    }
    for v in sorted_variants
], ensure_ascii=False)}

Crie uma resposta final que:
1. Consolide as melhores informações de todas as variantes
2. Resolva quaisquer contradições
3. Mantenha coerência e clareza
4. Seja mais abrangente do que qualquer variante única"""
        
        try:
            logger.debug(f"🔀 Merging {len(sorted_variants)} variants into single answer")
            response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=merge_prompt)]), timeout=60.0)
            answer = response.content.strip()
            # Remove thinking tags
            answer = answer.replace('<think>', '').replace('</think>', '').strip()
            logger.debug(f"✅ Variants merged successfully")
            return {
                'answer': answer,
                'sources': list(set(src for v in variants for src in v['sources']))
            }
        except asyncio.TimeoutError:
            logger.error(f"❌ Merge timeout, using highest-fitness variant")
            return {
                'answer': sorted_variants[0]['content'],
                'sources': sorted_variants[0]['sources']
            }
        except Exception as e:
            logger.error(f"❌ Merge failed: {e}")
            # Return best variant
            return {
                'answer': sorted_variants[0]['content'],
                'sources': sorted_variants[0]['sources']
            }
    
    async def refine_draft(self, state: TTDDRState) -> TTDDRState:
        """Report-level denoising with retrieval: Refine draft with new information"""
        logger.info(f"🔧 Report-level denoising (Iteration {state['current_iteration']}): Refining draft")
        self._update_progress(state['research_id'], 'refinement', f'Refining draft with new information from iteration {state["current_iteration"]}...')
        
        # Prepare answers from this iteration
        iteration_answers = [ans for ans in state['search_answers'] if ans['iteration'] == state['current_iteration']]
        
        if not iteration_answers:
            return state
        
        # Create answers summary
        answers_summary = "\n".join([
            f"Q: {ans['question']}\nA: {ans['answer']}\n"
            for ans in iteration_answers
        ])
        
        refinement_prompt = f"""Está a refinar um esboço de pesquisa com novas informações.

Esboço actual:
{state['draft']}

Novas informações da pesquisa:
{answers_summary}

Revise o esboço por:
1. Integrar as novas informações
2. Preencher lacunas identificadas no esboço actual
3. Fortalecer argumentos com evidência
4. Manter coerência e fluxo
5. Manter a estrutura intacta

Retorne o esboço refinado."""
        
        try:
            logger.debug(f"🔧 Refining draft at iteration {state['current_iteration']}")
            response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=refinement_prompt)]), timeout=60.0)
            draft = response.content.strip()
            # Remove thinking tags
            draft = draft.replace('<think>', '').replace('</think>', '').strip()
            state['draft'] = draft
            state['draft_history'].append(state['draft'])
            logger.info(f"✅ Draft refined at iteration {state['current_iteration']}")
            self._update_progress(state['research_id'], 'refinement', f'Draft successfully refined, {len(state["draft"])} characters')
        except asyncio.TimeoutError:
            logger.error(f"❌ Draft refinement timeout at iteration {state['current_iteration']}")
        except Exception as e:
            logger.error(f"❌ Draft refinement failed: {e}")
        
        return state
    
    async def check_iteration(self, state: TTDDRState) -> TTDDRState:
        """Check if we should continue iterating or move to final report"""
        state['current_iteration'] += 1
        logger.info(f"🔄 Completed iteration {state['current_iteration'] - 1}, max iterations: {state['max_iterations']}")
        return state
    
    async def generate_final_report(self, state: TTDDRState) -> TTDDRState:
        """Stage 3: Generate final polished report from all gathered information"""
        logger.info("🎓 Stage 3: Generating final report")
        self._update_progress(state['research_id'], 'final_report', 'Generating final polished research report from all gathered information...')
        
        # Prepare comprehensive context
        context_parts = [
            f"Research query: {state['query']}",
            f"Research plan key areas: {', '.join(state['research_plan']['key_areas'])}",
            "All questions and answers gathered:"
        ]
        
        for ans in state['search_answers']:
            context_parts.append(f"\nQ: {ans['question']}\nA: {ans['answer']}")
        
        context = "\n".join(context_parts)
        
        final_prompt = f"""Escreva um relatório de pesquisa final abrangente e bem estruturado com base em todas as informações recolhidas.

{context}

Crie um relatório final que:
1. Tenha secções claras organizadas por tópico
2. Sintetize informações de toda a pesquisa
3. Tire conclusões apoiadas por evidência
4. Identifique incertezas restantes
5. Sugira áreas para pesquisa futura
6. Seja polido e pronto para publicação"""
        
        try:
            logger.debug("🎓 Generating final report")
            self._update_progress(state['research_id'], 'final_report', 'Calling LLM to generate final polished report...')
            
            response = await asyncio.wait_for(self.llm.ainvoke([HumanMessage(content=final_prompt)]), timeout=60.0)
            report = response.content.strip()
            # Remove thinking tags
            report = report.replace('<think>', '').replace('</think>', '').strip()
            state['final_report'] = report
            logger.info("✅ Final report generated successfully")
            self._update_progress(state['research_id'], 'final_report', f'Final report completed! Total report length: {len(report)} characters')
            
        except asyncio.TimeoutError:
            logger.error("❌ Final report generation timeout, using draft")
            state['final_report'] = state['draft']
            self._update_progress(state['research_id'], 'final_report', 'Final report generation timed out, using refined draft instead')
            
        except Exception as e:
            logger.error(f"❌ Final report generation failed: {e}")
            state['final_report'] = state['draft']
            self._update_progress(state['research_id'], 'final_report', f'Final report generation failed: {str(e)}, using refined draft')
        
        # Calculate confidence based on number of iterations and answers
        total_answers = len(state['search_answers'])
        iterations_completed = state['current_iteration']
        
        # Confidence increases with more answers and iterations
        state['confidence_score'] = min(
            0.5 + (total_answers * 0.05) + (iterations_completed * 0.1),
            0.95
        )
        
        state['status'] = 'completed'
        logger.info(f"🏆 Final report generated. Confidence: {state['confidence_score']:.1%}")
        self._update_progress(state['research_id'], 'completion', f'Research completed with {total_answers} answers across {iterations_completed} iterations', confidence_score=state['confidence_score'])
        
        return state
    
    async def run(
        self,
        query: str,
        research_id: str,
        max_depth: int = 3,
        model_id: str = None
    ) -> dict:
        """Run the TTD-DR workflow"""
        logger.info(f"🚀 Starting TTD-DR: {research_id}")
        
        initial_state = TTDDRState(
            query=query,
            research_plan={'key_areas': [], 'outline': ''},
            draft='',
            draft_history=[],
            search_questions=[],
            search_answers=[],
            current_iteration=0,
            max_iterations=max_depth,
            final_report='',
            confidence_score=0.0,
            research_id=research_id,
            status='processing',
            sources_analyzed=0,
            all_variants_history=[]
        )
        
        try:
            result = await self.graph.ainvoke(initial_state)
            
            # Convert to API response format (compatible with existing interface)
            # Map search answers to thought_tree for compatibility
            thought_tree = [
                {
                    'id': f"answer_{i}",
                    'hypothesis': ans['question'],
                    'confidence': 0.8,
                    'supporting_evidence': [
                        {
                            'source': src,
                            'evidence': ans['answer'][:200],
                            'strength': 0.8
                        }
                        for src in ans['sources'][:3]
                    ],
                    'contradicting_evidence': [],
                    'status': 'proven',
                    'depth': ans['iteration'],
                    'sources_analyzed': 1
                }
                for i, ans in enumerate(result['search_answers'])
            ]
            
            return {
                'query': query,
                'synthesis': result['final_report'],
                'confidence_score': result['confidence_score'],
                'thought_tree': thought_tree,
                'best_path': [f"answer_{i}" for i in range(len(result['search_answers']))],
                'iterations': result['current_iteration'],
                'sources_count': result['sources_analyzed'],
                'status': 'completed'
            }
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Research failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'query': query,
                'synthesis': f"Research failed: {str(e)}",
                'confidence_score': 0.0,
                'thought_tree': [],
                'best_path': [],
                'iterations': 0,
                'sources_count': 0,
                'status': 'error'
            }
