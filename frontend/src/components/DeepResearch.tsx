import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { ChevronDown, Sparkles, Search, BookOpen } from 'lucide-react';

interface ModelOption {
  id: string;
  name: string;
  provider: string;
}

interface ResearchQuestion {
  id: string;
  hypothesis: string;
  confidence: number;
  sources_analyzed: number;
  depth: number;
}

interface DeepResearchResult {
  id: string;
  query: string;
  status: string;
  synthesis: string;
  confidence_score: number;
  sources_count: number;
  iterations: number;
  thought_tree: ResearchQuestion[];
  best_path: string[];
}

export const DeepResearch: React.FC<{ notebookId?: string }> = ({ notebookId }) => {
  const [query, setQuery] = useState('');
  const [modelId, setModelId] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [researchId, setResearchId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [currentStage, setCurrentStage] = useState<'plan' | 'search' | 'evolve' | 'refine' | 'final'>('plan');
  const [currentIteration, setCurrentIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(3);
  const [expandedQuestion, setExpandedQuestion] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('/api/deep-research/available-models');
        if (response.ok) {
          const data = await response.json();
          setAvailableModels(data.models);
          if (data.default_model_id) {
            setModelId(data.default_model_id);
          } else if (data.models.length > 0) {
            setModelId(data.models[0].id);
          }
        }
      } catch (error) {
        console.error('Failed to fetch models:', error);
      }
    };
    fetchModels();
  }, []);

  const startResearch = useMutation({
    mutationFn: async (searchQuery: string) => {
      const response = await fetch('/api/deep-research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          notebook_id: notebookId,
          save_to_notebook: true,
          max_iterations: maxIterations,
          model_id: modelId
        })
      });
      if (!response.ok) throw new Error('Failed to start research');
      return response.json();
    },
    onSuccess: (data) => {
      setResearchId(data.research_id);
      setIsStreaming(true);
      setCurrentIteration(0);
      setCurrentStage('plan');
      streamResearchProgress(data.research_id);
    },
    onError: (error) => {
      console.error('Failed to start research:', error);
    }
  });

  const streamResearchProgress = (id: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    eventSourceRef.current = new EventSource(`/api/deep-research/${id}/stream`);

    eventSourceRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgressMessage(data.message || '');
      
      // Update stage based on message content
      if (data.message?.includes('Stage 1')) setCurrentStage('plan');
      else if (data.message?.includes('Stage 2a')) setCurrentStage('search');
      else if (data.message?.includes('Stage 2b')) setCurrentStage('evolve');
      else if (data.message?.includes('Stage 2c') || data.message?.includes('denoising')) setCurrentStage('refine');
      else if (data.message?.includes('Stage 3')) setCurrentStage('final');
      
      // Extract iteration number from message
      const iterMatch = data.message?.match(/Iteration\s+(\d+)/);
      if (iterMatch) setCurrentIteration(parseInt(iterMatch[1]));

      if (data.status === 'completed' || data.status === 'failed') {
        setIsStreaming(false);
        eventSourceRef.current?.close();
        refetchResult();
      }
    };

    eventSourceRef.current.onerror = () => {
      setIsStreaming(false);
      eventSourceRef.current?.close();
    };
  };

  const { data: result, refetch: refetchResult } = useQuery({
    queryKey: ['deepResearch', researchId],
    queryFn: async () => {
      const response = await fetch(`/api/deep-research/${researchId}/full`);
      if (!response.ok) throw new Error('Not ready yet');
      return response.json() as Promise<DeepResearchResult>;
    },
    enabled: !!researchId,
    refetchInterval: isStreaming ? 1000 : false
  });

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const StageIndicator = () => {
    const stages: Array<{ key: 'plan' | 'search' | 'evolve' | 'refine' | 'final'; label: string }> = [
      { key: 'plan', label: 'Research Plan' },
      { key: 'search', label: 'Search Questions' },
      { key: 'evolve', label: 'Self-Evolution' },
      { key: 'refine', label: 'Draft Refinement' },
      { key: 'final', label: 'Final Report' }
    ];

    return (
      <div className="flex items-center gap-1 mb-6">
        {stages.map((stage, idx) => (
          <React.Fragment key={stage.key}>
            <div
              className={`flex flex-col items-center gap-1 transition-all ${
                currentStage === stage.key ? 'scale-110' : ''
              }`}
            >
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold transition-all ${
                  currentStage === stage.key
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white'
                    : stages.findIndex(s => s.key === currentStage) > idx
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-300 text-gray-700'
                }`}
              >
                {stages.findIndex(s => s.key === currentStage) > idx ? '✓' : idx + 1}
              </div>
              <span className="text-xs font-medium text-gray-600 text-center">{stage.label}</span>
            </div>
            {idx < stages.length - 1 && (
              <div className={`flex-1 h-0.5 ${stages.findIndex(s => s.key === currentStage) > idx ? 'bg-green-500' : 'bg-gray-300'}`} />
            )}
          </React.Fragment>
        ))}
      </div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl shadow-2xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-4xl font-bold text-white">Deep Research</h2>
        </div>
        <p className="text-slate-400">Advanced research through iterative refinement and self-evolution</p>
      </div>

      {/* Input Section */}
      <div className="mb-8 p-6 bg-slate-800 rounded-xl border border-slate-700">
        <label className="block text-sm font-semibold text-slate-300 mb-3">
          What do you want to research?
        </label>
        <div className="flex gap-2 mb-6">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && startResearch.mutate(query)}
            placeholder="Enter your research query ..."
            className="flex-1 px-4 py-3 bg-slate-700 border border-slate-600 text-white placeholder-slate-400 rounded-lg focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition"
            disabled={startResearch.isPending || isStreaming || !modelId}
          />
          <button
            onClick={() => startResearch.mutate(query)}
            disabled={!query || startResearch.isPending || isStreaming || !modelId}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold rounded-lg hover:shadow-xl hover:shadow-cyan-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            {startResearch.isPending || isStreaming ? 'Researching...' : 'Start Research'}
          </button>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Model Selection
            </label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={startResearch.isPending || isStreaming}
              className="w-full px-4 py-2 bg-slate-700 border border-slate-600 text-white rounded-lg focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition disabled:opacity-50"
            >
              <option value="">
                {availableModels.length === 0 ? 'Loading models...' : 'Select a model'}
              </option>
              {availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.provider})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Max Iterations
            </label>
            <select
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value))}
              disabled={startResearch.isPending || isStreaming}
              className="w-full px-4 py-2 bg-slate-700 border border-slate-600 text-white rounded-lg focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition disabled:opacity-50"
            >
              <option value="1">1 Iteration</option>
              <option value="2">2 Iterations</option>
              <option value="3">3 Iterations</option>
              <option value="4">4 Iterations</option>
              <option value="5">5 Iterations</option>
            </select>
          </div>
        </div>
      </div>

      {/* Progress Section */}
      {isStreaming && (
        <div className="mb-8 space-y-6">
          <StageIndicator />
          
          <div className="p-6 bg-gradient-to-r from-blue-900 to-cyan-900 rounded-xl border border-cyan-500 shadow-lg shadow-cyan-500/20">
            <div className="flex items-center gap-3 mb-3">
              <div className="flex items-center justify-center w-6 h-6 bg-cyan-500 rounded-full animate-spin">
                <div className="w-3 h-3 bg-blue-900 rounded-full"></div>
              </div>
              <p className="text-sm font-bold text-cyan-300">Processing...</p>
            </div>
            <p className="text-sm text-cyan-100 mb-4">{progressMessage}</p>
            {currentIteration > 0 && (
              <p className="text-xs text-cyan-300">Iteration {currentIteration} of {maxIterations}</p>
            )}
            <div className="mt-4 w-full bg-cyan-900 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-500 to-cyan-400 h-2 rounded-full animate-pulse"
                style={{ width: `${(currentIteration / maxIterations) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Main Synthesis */}
          <div className="p-8 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl border border-slate-600">
            <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
              <BookOpen className="w-6 h-6 text-cyan-400" />
              Final Research Report
            </h3>
            <div className="prose prose-invert max-w-none">
              <p className="text-slate-200 leading-relaxed whitespace-pre-wrap text-sm">
                {result.synthesis}
              </p>
            </div>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-6 bg-gradient-to-br from-blue-900 to-blue-800 rounded-xl border border-blue-600 shadow-lg">
              <h4 className="text-xs font-bold text-blue-300 uppercase tracking-wider">Confidence Score</h4>
              <p className="text-4xl font-bold text-blue-100 mt-2">
                {Math.round(result.confidence_score * 100)}%
              </p>
              <div className="mt-3 w-full bg-blue-900 rounded-full h-1.5">
                <div
                  className="bg-blue-400 h-1.5 rounded-full"
                  style={{ width: `${result.confidence_score * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="p-6 bg-gradient-to-br from-green-900 to-green-800 rounded-xl border border-green-600 shadow-lg">
              <h4 className="text-xs font-bold text-green-300 uppercase tracking-wider">Sources Analyzed</h4>
              <p className="text-4xl font-bold text-green-100 mt-2">{result.sources_count}</p>
              <p className="text-xs text-green-400 mt-2">documents reviewed</p>
            </div>

            <div className="p-6 bg-gradient-to-br from-purple-900 to-purple-800 rounded-xl border border-purple-600 shadow-lg">
              <h4 className="text-xs font-bold text-purple-300 uppercase tracking-wider">Iterations</h4>
              <p className="text-4xl font-bold text-purple-100 mt-2">{result.iterations}</p>
              <p className="text-xs text-purple-400 mt-2">refinement cycles</p>
            </div>

            <div className="p-6 bg-gradient-to-br from-orange-900 to-orange-800 rounded-xl border border-orange-600 shadow-lg">
              <h4 className="text-xs font-bold text-orange-300 uppercase tracking-wider">Questions Asked</h4>
              <p className="text-4xl font-bold text-orange-100 mt-2">{result.thought_tree?.length || 0}</p>
              <p className="text-xs text-orange-400 mt-2">research queries</p>
            </div>
          </div>

          {/* Research Questions and Answers */}
          {result.thought_tree && result.thought_tree.length > 0 && (
            <div className="p-8 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl border border-slate-600">
              <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Search className="w-6 h-6 text-cyan-400" />
                Research Questions & Evolution
              </h3>
              <div className="space-y-4">
                {result.thought_tree.map((item, idx) => (
                  <div
                    key={item.id || idx}
                    className={`p-5 rounded-lg border-2 cursor-pointer transition-all ${
                      expandedQuestion === item.id
                        ? 'bg-slate-600 border-cyan-500 shadow-lg shadow-cyan-500/20'
                        : 'bg-slate-700 border-slate-600 hover:border-slate-500'
                    }`}
                    onClick={() =>
                      setExpandedQuestion(expandedQuestion === item.id ? null : item.id)
                    }
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <h4 className="text-white font-semibold mb-2">{item.hypothesis}</h4>
                        <div className="flex flex-wrap gap-3 text-xs text-slate-400">
                          <span className="flex items-center gap-1">
                            <span className="text-cyan-400">•</span>
                            Confidence: <span className="text-cyan-300 font-bold">{Math.round(item.confidence * 100)}%</span>
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="text-green-400">•</span>
                            Sources: <span className="text-green-300 font-bold">{item.sources_analyzed}</span>
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="text-orange-400">•</span>
                            Iteration: <span className="text-orange-300 font-bold">{item.depth}</span>
                          </span>
                        </div>
                      </div>
                      <ChevronDown
                        className={`w-5 h-5 text-slate-400 transition-transform flex-shrink-0 ${
                          expandedQuestion === item.id ? 'rotate-180' : ''
                        }`}
                      />
                    </div>

                    {expandedQuestion === item.id && (
                      <div className="mt-5 pt-5 border-t border-slate-600 space-y-4">
                        <div>
                          <h5 className="text-sm font-semibold text-slate-300 mb-2">Answer Summary</h5>
                          <p className="text-sm text-slate-300 leading-relaxed">
                            {typeof item.hypothesis === 'string' ? item.hypothesis : 'Research conducted on this topic'}
                          </p>
                        </div>
                        {item.sources_analyzed > 0 && (
                          <div>
                            <h5 className="text-sm font-semibold text-slate-300 mb-2">Sources Used</h5>
                            <p className="text-sm text-slate-400">{item.sources_analyzed} sources integrated in synthesis</p>
                          </div>
                        )}
                        <div className="flex gap-2 pt-2">
                          <div className="flex-1 bg-slate-600 rounded-lg p-3">
                            <p className="text-xs text-slate-400 font-medium">Confidence</p>
                            <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
                              <div
                                className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full"
                                style={{ width: `${item.confidence * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
