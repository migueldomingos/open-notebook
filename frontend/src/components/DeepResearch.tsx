import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { ChevronDown } from 'lucide-react';

interface BranchNode {
  id: string;
  hypothesis: string;
  confidence: number;
  status: 'exploring' | 'promising' | 'rejected' | 'proven' | 'inconclusive' | 'error';
  depth: number;
  sources_analyzed: number;
}

interface ModelOption {
  id: string;
  name: string;
  provider: string;
}

interface DeepResearchResult {
  id: string;
  query: string;
  status: string;
  synthesis: string;
  confidence_score: number;
  sources_count: number;
  iterations: number;
  thought_tree: BranchNode[];
  best_path: string[];
}

export const DeepResearch: React.FC<{ notebookId?: string }> = ({ notebookId }) => {
  const [query, setQuery] = useState('');
  const [modelId, setModelId] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [researchId, setResearchId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [expandedBranch, setExpandedBranch] = useState<string | null>(null);
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
          max_iterations: 3,
          model_id: modelId
        })
      });
      if (!response.ok) throw new Error('Failed to start research');
      return response.json();
    },
    onSuccess: (data) => {
      setResearchId(data.research_id);
      setIsStreaming(true);
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'proven':
      case 'promising':
        return 'bg-green-50 border-green-200 text-green-900';
      case 'rejected':
      case 'error':
        return 'bg-red-50 border-red-200 text-red-900';
      case 'exploring':
        return 'bg-blue-50 border-blue-200 text-blue-900';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-900';
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'proven':
      case 'promising':
        return <span className="px-2 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded">Promising</span>;
      case 'rejected':
      case 'error':
        return <span className="px-2 py-1 text-xs font-semibold bg-red-100 text-red-800 rounded">Rejected</span>;
      case 'exploring':
        return <span className="px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-800 rounded animate-pulse">Exploring</span>;
      default:
        return <span className="px-2 py-1 text-xs font-semibold bg-gray-100 text-gray-800 rounded">{status}</span>;
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-3xl font-bold mb-6 text-gray-900">Deep Research</h2>

      {/* Input Section */}
      <div className="mb-8">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Research Query
        </label>
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && startResearch.mutate(query)}
            placeholder="Enter your research query (e.g., 'How does quantum computing work?')..."
            className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none transition"
            disabled={startResearch.isPending || isStreaming || !modelId}
          />
          <button
            onClick={() => startResearch.mutate(query)}
            disabled={!query || startResearch.isPending || isStreaming || !modelId}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-medium rounded-lg hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {startResearch.isPending || isStreaming ? 'Researching...' : 'Start Deep Research'}
          </button>
        </div>
        
        <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">
          Model Selection
        </label>
        <select
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          disabled={startResearch.isPending || isStreaming}
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none transition disabled:opacity-50"
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

      {/* Progress Section */}
      {isStreaming && (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-cyan-50 border-l-4 border-blue-500 rounded-lg">
          <p className="text-sm font-medium text-blue-900 mb-2">Research in progress...</p>
          <p className="text-sm text-blue-700">{progressMessage}</p>
          <div className="mt-3 w-full bg-blue-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full animate-pulse"
              style={{ width: '70%' }}
            ></div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Synthesis */}
          <div className="p-6 bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg border border-gray-200">
            <h3 className="text-xl font-bold mb-3 text-gray-900">Synthesis</h3>
            <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
              {result.synthesis}
            </p>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h4 className="text-xs font-semibold text-blue-600 uppercase">Confidence</h4>
              <p className="text-3xl font-bold text-blue-700 mt-1">
                {Math.round(result.confidence_score * 100)}%
              </p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
              <h4 className="text-xs font-semibold text-green-600 uppercase">Sources</h4>
              <p className="text-3xl font-bold text-green-700 mt-1">{result.sources_count}</p>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
              <h4 className="text-xs font-semibold text-purple-600 uppercase">Iterations</h4>
              <p className="text-3xl font-bold text-purple-700 mt-1">{result.iterations}</p>
            </div>
            <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
              <h4 className="text-xs font-semibold text-orange-600 uppercase">Branches</h4>
              <p className="text-3xl font-bold text-orange-700 mt-1">
                {result.thought_tree?.length || 0}
              </p>
            </div>
          </div>

          {/* Thought Tree Visualization */}
          {result.thought_tree && result.thought_tree.length > 0 && (
            <div className="p-6 bg-white rounded-lg border border-gray-200">
              <h3 className="text-lg font-bold mb-4 text-gray-900">Research Branches</h3>
              <div className="space-y-3">
                {result.thought_tree.map((branch) => (
                  <div
                    key={branch.id}
                    className={`p-4 border-2 rounded-lg cursor-pointer transition ${getStatusColor(
                      branch.status
                    )}`}
                    onClick={() =>
                      setExpandedBranch(
                        expandedBranch === branch.id ? null : branch.id
                      )
                    }
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3 flex-1">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h4 className="font-semibold flex-1">
                              {branch.hypothesis}
                            </h4>
                            {getStatusBadge(branch.status)}
                          </div>
                          <div className="flex gap-4 text-xs opacity-75">
                            <span>
                              Confidence:{' '}
                              <span className="font-bold">
                                {Math.round(branch.confidence * 100)}%
                              </span>
                            </span>
                            <span>
                              Sources:{' '}
                              <span className="font-bold">{branch.sources_analyzed}</span>
                            </span>
                          </div>
                        </div>
                      </div>
                      <ChevronDown
                        className={`w-5 h-5 transition ${
                          expandedBranch === branch.id ? 'transform rotate-180' : ''
                        }`}
                      />
                    </div>

                    {/* Expanded details */}
                    {expandedBranch === branch.id && (
                      <div className="mt-4 pt-4 border-t-2 border-current border-opacity-20">
                        <div className="space-y-4">
                          <div>
                            <h5 className="text-sm font-semibold mb-2">Confidence Progression</h5>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full"
                                style={{
                                  width: `${Math.round(branch.confidence * 100)}%`
                                }}
                              ></div>
                            </div>
                          </div>
                          <div>
                            <h5 className="text-sm font-semibold mb-2">Exploration Depth</h5>
                            <p className="text-sm opacity-75">Depth: {branch.depth}</p>
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
