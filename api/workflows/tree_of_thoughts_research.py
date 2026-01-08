from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
import json
import asyncio
from datetime import datetime
import logging
from open_notebook.database.repository import repo_query

logger = logging.getLogger(__name__)

class ThoughtNode(TypedDict):
    id: str
    hypothesis: str
    confidence: float
    supporting_evidence: List[dict]
    contradicting_evidence: List[dict]
    status: str  # "exploring", "promising", "rejected", "proven", "inconclusive"
    depth: int
    sources_analyzed: int

class ToTResearchState(TypedDict):
    query: str
    thought_tree: List[ThoughtNode]
    active_branches: List[str]  # ids of branches to explore
    best_path: List[str]
    synthesis: str
    confidence_score: float
    max_depth: int
    iteration_count: int
    research_id: str
    status: str

class TreeOfThoughtsResearch:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(ToTResearchState)
        
        workflow.add_node("generate_initial_thoughts", self.generate_initial_thoughts)
        workflow.add_node("expand_branches", self.expand_all_branches)
        workflow.add_node("evaluate_branches", self.evaluate_all_branches)
        workflow.add_node("prune_branches", self.prune_weak_branches)
        workflow.add_node("synthesize", self.synthesize_best_path)
        workflow.add_node("confidence_assessment", self.assess_confidence)
        
        workflow.add_edge("generate_initial_thoughts", "expand_branches")
        workflow.add_edge("expand_branches", "evaluate_branches")
        workflow.add_edge("evaluate_branches", "prune_branches")
        workflow.add_conditional_edges(
            "prune_branches",
            self.should_continue_exploring,
            {
                "continue": "expand_branches",
                "synthesize": "synthesize"
            }
        )
        workflow.add_edge("synthesize", "confidence_assessment")
        workflow.add_edge("confidence_assessment", END)
        
        workflow.set_entry_point("generate_initial_thoughts")
        return workflow.compile()
    
    async def generate_initial_thoughts(self, state: ToTResearchState) -> ToTResearchState:
        """Generate initial hypothesis branches using Tree-of-Thoughts"""
        logger.info(f"💡 Generating initial thoughts for query: {state['query']}")
        
        prompt = f"""Gere 5 hipóteses diversas e testáveis para investigar a pergunta: "{state['query']}"

Cada hipótese deve ser:
- Distinta e explorar ângulos diferentes
- Testável contra evidências disponíveis
- Fundamentada em suposições razoáveis

Retorne APENAS como array JSON com campos:
- hypothesis: texto da hipótese (máx 100 caracteres)
- reasoning: breve explicação de por que é plausível

Exemplo de formato:
[
  {{"hypothesis": "...", "reasoning": "..."}},
  {{"hypothesis": "...", "reasoning": "..."}}
]

Retorne apenas o JSON, sem explicações adicionais."""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Extract JSON if wrapped in markdown code blocks
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            logger.debug(f"LLM response: {response_text}")
            hypotheses = json.loads(response_text)
            
            if not isinstance(hypotheses, list):
                hypotheses = [hypotheses] if isinstance(hypotheses, dict) else []
            
            state['thought_tree'] = [
                ThoughtNode(
                    id=f"branch_{i}",
                    hypothesis=h.get('hypothesis', ''),
                    confidence=0.5,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    status="exploring",
                    depth=0,
                    sources_analyzed=0
                ) for i, h in enumerate(hypotheses) if h.get('hypothesis')
            ]
            state['active_branches'] = [f"branch_{i}" for i in range(len(state['thought_tree']))]
            state['iteration_count'] = 0
            logger.info(f"✅ Generated {len(state['thought_tree'])} initial hypotheses")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"❌ Failed to parse hypotheses: {e}")
            logger.error(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
            # Fallback to default hypotheses
            state['thought_tree'] = [
                ThoughtNode(
                    id=f"branch_{i}",
                    hypothesis=f"Hypothesis {i+1} about {state['query']}",
                    confidence=0.3,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    status="exploring",
                    depth=0,
                    sources_analyzed=0
                ) for i in range(3)
            ]
            state['active_branches'] = [f"branch_{i}" for i in range(3)]
            state['iteration_count'] = 0
        except Exception as e:
            logger.exception(f"Unexpected error in generate_initial_thoughts: {e}")
            state['status'] = "failed"
        
        return state
    
    async def expand_all_branches(self, state: ToTResearchState) -> ToTResearchState:
        """Expand all active branches in parallel"""
        logger.info(f"🌲 Expanding {len(state['active_branches'])} branches at iteration {state['iteration_count']}")
        
        tasks = []
        for branch_id in state['active_branches']:
            branch_idx = next(i for i, b in enumerate(state['thought_tree']) if b['id'] == branch_id)
            tasks.append(self._expand_single_branch(state, branch_idx))
        
        updated_branches = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, updated in enumerate(updated_branches):
            if not isinstance(updated, Exception):
                branch_idx = next(j for j, b in enumerate(state['thought_tree']) if b['id'] == state['active_branches'][i])
                state['thought_tree'][branch_idx] = updated
        
        state['iteration_count'] += 1
        return state
    
    async def _expand_single_branch(self, state: ToTResearchState, branch_idx: int) -> ThoughtNode:
        """Expand a single branch with ReAct loop (Reasoning + Acting)"""
        branch = state['thought_tree'][branch_idx]
        
        if branch['depth'] >= state['max_depth']:
            return branch
        
        logger.info(f"🌳 Expanding branch {branch['id']}: {branch['hypothesis'][:50]}...")
        
        try:
            # STEP 1: REASONING - What evidence would support/refute this?
            reasoning_prompt = f"""Hipótese: {branch['hypothesis']}
Contagem de evidências de suporte atual: {len(branch['supporting_evidence'])}
Contagem de evidências contradutórias atuais: {len(branch['contradicting_evidence'])}

Com base nesta hipótese, que 3-4 consultas de busca específicas testariam melhor esta hipótese?
Foco em encontrar evidências que poderiam SUPORTAR OU REFUTAR.

Retorne APENAS como JSON, sem explicações:
{{
  "queries": ["consulta1", "consulta2", "consulta3"],
  "search_strategy": "breve explicação"
}}"""
            
            reasoning = await self.llm.ainvoke([HumanMessage(content=reasoning_prompt)])
            reasoning_text = reasoning.content.strip()
            if '```json' in reasoning_text:
                reasoning_text = reasoning_text.split('```json')[1].split('```')[0].strip()
            elif '```' in reasoning_text:
                reasoning_text = reasoning_text.split('```')[1].split('```')[0].strip()
            
            reasoning_result = json.loads(reasoning_text)
            search_queries = reasoning_result.get('queries', [])
            
            if not search_queries:
                search_queries = [branch['hypothesis']]
            
            # STEP 2: ACTING - Search for evidence
            logger.info(f"🔍 Searching for evidence with {len(search_queries)} queries")
            new_sources = await self._multi_query_search(search_queries)
            branch['sources_analyzed'] += len(new_sources)
            
            # STEP 3: OBSERVING - Analyze findings
            if new_sources:
                # Build sources data outside f-string to avoid brace conflicts
                sources_data = json.dumps([{'content': s.page_content[:200], 'source': s.metadata.get('source', 'desconhecido')} for s in new_sources[:10]])
                analysis_prompt = f"""Analise estas fontes no contexto da hipótese: "{branch['hypothesis']}"

Fontes a analisar:
{sources_data}

Para cada fonte, determine se:
1. Suporta a hipótese (supporting_evidence)
2. Contradiz a hipótese (contradicting_evidence)
3. É neutra/irrelevante (ignorar)

Retorne APENAS como JSON:
{{
  "supporting_evidence": [
    {{"source": "...", "evidence": "...", "strength": 0.7}},
  ],
  "contradicting_evidence": []
}}"""
                
                analysis = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
                analysis_text = analysis.content.strip()
                if '```json' in analysis_text:
                    analysis_text = analysis_text.split('```json')[1].split('```')[0].strip()
                elif '```' in analysis_text:
                    analysis_text = analysis_text.split('```')[1].split('```')[0].strip()
                
                findings = json.loads(analysis_text)
                
                # Ensure evidence is properly structured (convert any nested dicts to ensure they're JSON-serializable)
                supporting = findings.get('supporting_evidence', [])
                contradicting = findings.get('contradicting_evidence', [])
                
                # Validate and clean evidence data
                if isinstance(supporting, list):
                    # Convert each evidence item to a clean dict with string values
                    supporting = [
                        {
                            'source': str(e.get('source', '')) if isinstance(e, dict) else '',
                            'evidence': str(e.get('evidence', '')) if isinstance(e, dict) else '',
                            'strength': float(e.get('strength', 0.7)) if isinstance(e, dict) else 0.7
                        }
                        for e in supporting
                        if isinstance(e, dict) and e.get('evidence')
                    ]
                
                if isinstance(contradicting, list):
                    # Convert each evidence item to a clean dict with string values
                    contradicting = [
                        {
                            'source': str(e.get('source', '')) if isinstance(e, dict) else '',
                            'evidence': str(e.get('evidence', '')) if isinstance(e, dict) else '',
                            'strength': float(e.get('strength', 0.7)) if isinstance(e, dict) else 0.7
                        }
                        for e in contradicting
                        if isinstance(e, dict) and e.get('evidence')
                    ]
                
                # Update branch with new evidence
                branch['supporting_evidence'].extend(supporting)
                branch['contradicting_evidence'].extend(contradicting)
            
            branch['depth'] += 1
            
            # Calculate confidence based on evidence ratio
            total_evidence = len(branch['supporting_evidence']) + len(branch['contradicting_evidence'])
            if total_evidence > 0:
                supporting_sum = sum(e.get('strength', 0.7) for e in branch['supporting_evidence'][:5])
                contradicting_sum = sum(e.get('strength', 0.7) for e in branch['contradicting_evidence'][:5])
                branch['confidence'] = supporting_sum / (supporting_sum + contradicting_sum + 0.1)
            else:
                branch['confidence'] = 0.5
            
            # Determine branch status
            if branch['confidence'] < 0.25:
                branch['status'] = "rejected"
            elif branch['confidence'] > 0.75:
                branch['status'] = "promising"
            else:
                branch['status'] = "exploring"
            
            logger.info(f"✅ Branch {branch['id']} updated: confidence={branch['confidence']:.2f}, status={branch['status']}")
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"❌ Error expanding branch {branch['id']}: {e}")
            branch['status'] = "error"
            branch['confidence'] = 0.3
        except Exception as e:
            logger.exception(f"Unexpected error expanding branch {branch['id']}: {e}")
            branch['status'] = "error"
        
        return branch
    
    async def evaluate_all_branches(self, state: ToTResearchState) -> ToTResearchState:
        """Evaluate and compare all branches"""
        logger.info("⚖️  Evaluating all branches")
        
        # Build branches data outside f-string to avoid brace conflicts
        branches_data = json.dumps([
            {
                'hypothesis': b['hypothesis'],
                'confidence': b['confidence'],
                'supporting_count': len(b['supporting_evidence']),
                'contradicting_count': len(b['contradicting_evidence']),
                'status': b['status']
            } for b in state['thought_tree']
        ])
        evaluation_prompt = f"""Avalie estes ramos de pesquisa:

{branches_data}

Forneça análise:
1. Quais hipóteses são mais promissoras?
2. Existem contradições que precisam ser resolvidas?
3. Qual é a conclusão geral mais forte?

Retorne APENAS como JSON com análise comparativa."""
        
        try:
            evaluation = await self.llm.ainvoke([HumanMessage(content=evaluation_prompt)])
            evaluation_text = evaluation.content.strip()
            if '```json' in evaluation_text:
                evaluation_text = evaluation_text.split('```json')[1].split('```')[0].strip()
            elif '```' in evaluation_text:
                evaluation_text = evaluation_text.split('```')[1].split('```')[0].strip()
            
            state['evaluation'] = json.loads(evaluation_text)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"❌ Error evaluating branches: {e}")
            state['evaluation'] = {"analysis": "Evaluation in progress"}
        except Exception as e:
            logger.exception(f"Unexpected error evaluating branches: {e}")
            state['evaluation'] = {"analysis": "Evaluation in progress"}
        
        return state
    
    async def prune_weak_branches(self, state: ToTResearchState) -> ToTResearchState:
        """Remove branches that are unlikely to yield useful insights"""
        logger.info("✂️  Pruning weak branches")
        
        state['active_branches'] = [
            b['id'] for b in state['thought_tree']
            if b['status'] not in ['rejected', 'error'] and b['depth'] < state['max_depth']
        ]
        
        logger.info(f"✅ Active branches remaining: {len(state['active_branches'])}")
        return state
    
    def should_continue_exploring(self, state: ToTResearchState) -> str:
        """Decide whether to continue exploring or synthesize"""
        # Continue if we have active branches and haven't hit max depth
        if state['active_branches'] and state['iteration_count'] < state['max_depth']:
            # Check if any branch is still exploring
            has_exploring = any(
                b['status'] == 'exploring' for b in state['thought_tree']
                if b['id'] in state['active_branches']
            )
            if has_exploring:
                return "continue"
        
        return "synthesize"
    
    async def synthesize_best_path(self, state: ToTResearchState) -> ToTResearchState:
        """Synthesize findings from best-supported branches"""
        logger.info("🎨 Synthesizing research findings")
        
        # Rank branches by confidence
        ranked_branches = sorted(
            enumerate(state['thought_tree']),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        top_branches = ranked_branches[:3]
        state['best_path'] = [b[1]['id'] for b in top_branches]
        
        # Prepare synthesis data
        synthesis_data = [{
            'rank': i + 1,
            'hypothesis': b[1]['hypothesis'],
            'confidence': round(b[1]['confidence'], 2),
            'supporting_evidence': b[1]['supporting_evidence'][:5],
            'contradicting_evidence': b[1]['contradicting_evidence'][:3],
            'status': b[1]['status']
        } for i, b in enumerate(top_branches)]
        
        synthesis_prompt = f"""Sintetize as descobertas abrangentes de pesquisa com base na avaliação de hipóteses:

Hipóteses Principais:
{json.dumps(synthesis_data, indent=2)}

Pergunta Original: {state['query']}

Crie uma síntese detalhada que:
1. Identifique a hipótese mais apoiada e por quê
2. Reconheça perspectivas válidas concorrentes
3. Destaque evidência-chave e fontes
4. Identifique incertezas remanescentes
5. Sugira áreas para pesquisa adicional

Forneça uma resposta coerente e bem fundamentada adequada para um caderno de pesquisa."""
        
        try:
            synthesis = await self.llm.ainvoke([HumanMessage(content=synthesis_prompt)])
            state['synthesis'] = synthesis.content
            logger.info("✅ Synthesis completed")
        except Exception as e:
            logger.error(f"❌ Error synthesizing: {e}")
            state['synthesis'] = f"Synthesis generated from {len(state['thought_tree'])} explored branches with top confidence: {max((b['confidence'] for b in state['thought_tree']), default=0):.1%}"
        
        return state
    
    async def assess_confidence(self, state: ToTResearchState) -> ToTResearchState:
        """Assess overall confidence in findings"""
        logger.info("🏆 Assessing research confidence")
        
        # Calculate metrics
        total_branches = len(state['thought_tree'])
        promising_branches = sum(1 for b in state['thought_tree'] if b['status'] in ['proven', 'promising'])
        avg_confidence = sum(b['confidence'] for b in state['thought_tree']) / total_branches if total_branches > 0 else 0
        total_evidence = sum(len(b['supporting_evidence']) for b in state['thought_tree'])
        
        confidence_prompt = f"""Avalie a confiança nestes achados de pesquisa:

Métricas:
- Ramos explorados: {total_branches}
- Ramos com status promissor/comprovado: {promising_branches}
- Confiança média do ramo: {avg_confidence:.2f}
- Total de peças de evidências encontradas: {total_evidence}
- Iterações concluídas: {state['iteration_count']}

Resumo da síntese: {state['synthesis'][:500]}

Forneça APENAS JSON com:
{{
  "overall_confidence": 75,
  "confidence_factors": ["fator1", "fator2"],
  "uncertainty_areas": ["área1"],
  "reliability_assessment": "alto",
  "reasoning": "breve"
}}"""
        
        try:
            confidence_result = await self.llm.ainvoke([HumanMessage(content=confidence_prompt)])
            confidence_text = confidence_result.content.strip()
            if '```json' in confidence_text:
                confidence_text = confidence_text.split('```json')[1].split('```')[0].strip()
            elif '```' in confidence_text:
                confidence_text = confidence_text.split('```')[1].split('```')[0].strip()
            
            confidence_data = json.loads(confidence_text)
            state['confidence_score'] = confidence_data.get('overall_confidence', 50) / 100.0
            state['confidence_data'] = confidence_data
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"❌ Error assessing confidence: {e}")
            # Calculate confidence from branch metrics
            state['confidence_score'] = min(0.95, avg_confidence * 0.7 + (promising_branches / max(total_branches, 1)) * 0.3)
            state['confidence_data'] = {
                'overall_confidence': int(state['confidence_score'] * 100),
                'reasoning': f'Calculated from {total_branches} branches with {total_evidence} evidence pieces'
            }
        except Exception as e:
            logger.exception(f"Unexpected error assessing confidence: {e}")
            state['confidence_score'] = 0.5
            state['confidence_data'] = {'overall_confidence': 50, 'reasoning': 'Assessment in progress'}
        
        state['status'] = "completed"
        logger.info(f"✅ Research completed with confidence score: {state['confidence_score']:.2f}")
        return state
    
    async def _multi_query_search(self, queries: List[str]) -> List[Document]:
        """Search with multiple refined queries using embeddings"""
        results = []
        seen_ids = set()
        
        for query in queries[:4]:  # Limit to 4 queries per iteration
            try:
                # Generate embedding for the query
                query_embedding = (await self.embeddings.aembed([query]))[0]
                
                # Search using SurrealDB vector search function
                search_results = await repo_query(
                    """
                    SELECT * FROM fn::vector_search($embed, $results, true, false, $minimum_score);
                    """,
                    {
                        "embed": query_embedding,
                        "results": 8,
                        "minimum_score": 0.3,  # Only results with similarity > 0.3
                    },
                )
                
                if not isinstance(search_results, list):
                    logger.warning(f"Search results is not a list: {type(search_results)}")
                    search_results = list(search_results) if search_results else []
                
                for source in search_results:
                    try:
                        # Ensure source is a dict
                        if not isinstance(source, dict):
                            logger.warning(f"Source is not a dict: {type(source)}, skipping")
                            continue
                        
                        # Ensure source_id is a string (convert from RecordID or dict if needed)
                        source_id = source.get('id')
                        
                        # Convert to string and ensure it's hashable
                        if source_id is None:
                            source_id = str(id(source))
                        else:
                            # Force conversion to string, handling any type
                            source_id = str(source_id)
                        
                        # Check if we've already seen this source
                        if source_id not in seen_ids:
                            # Convert source data to Document format
                            doc = Document(
                                page_content=source.get('full_text', source.get('content', '')),
                                metadata={
                                    'id': source_id,
                                    'title': source.get('title', ''),
                                    'source': source_id,
                                    'relevance': source.get('score', 0)  # Add relevance score
                                }
                            )
                            results.append(doc)
                            seen_ids.add(source_id)
                    except TypeError as te:
                        logger.error(f"Type error processing source: {te}, source_id type: {type(source_id)}, source_id value: {source_id}")
                        continue
            except Exception as e:
                logger.error(f"Error searching for query '{query}': {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return results
    
    async def run(self, query: str, research_id: str = None, max_depth: int = 3, model_id: str = None):
        """Execute the Tree-of-Thoughts research workflow"""
        logger.info(f"🚀 Starting ToT research for query: {query} with model_id: {model_id}")
        
        initial_state = ToTResearchState(
            query=query,
            thought_tree=[],
            active_branches=[],
            best_path=[],
            synthesis="",
            confidence_score=0.0,
            max_depth=max_depth,
            iteration_count=0,
            research_id=research_id or "unknown",
            status="processing"
        )
        
        try:
            result = await self.graph.ainvoke(initial_state)
            return {
                'query': query,
                'synthesis': result['synthesis'],
                'confidence_score': result['confidence_score'],
                'thought_tree': result['thought_tree'],
                'best_path': result['best_path'],
                'iterations': result['iteration_count'],
                'sources_count': sum(b['sources_analyzed'] for b in result['thought_tree']),
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
                'status': 'failed'
            }
