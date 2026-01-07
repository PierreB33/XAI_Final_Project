import * as React from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { Upload, Zap, Eye, BarChart3, AlertCircle, CheckCircle2 } from 'lucide-react'
import { apiService, type AnalysisResponse, type Model, type XAITechnique } from '~/services/api'

//@ts-ignore
export const Route = createFileRoute('/')({
  component: XAIAnalysis,
})

type MediaType = 'audio' | 'image' | null
type Status = 'idle' | 'loading' | 'success' | 'error'

function XAIAnalysis() {
  const [mediaType, setMediaType] = React.useState<MediaType>(null)
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null)
  const [selectedModel, setSelectedModel] = React.useState<string | null>(null)
  const [selectedXAI, setSelectedXAI] = React.useState<string[]>([])
  const [status, setStatus] = React.useState<Status>('idle')
  const [error, setError] = React.useState<string | null>(null)
  const [results, setResults] = React.useState<AnalysisResponse | null>(null)
  const [audioModels, setAudioModels] = React.useState<Model[]>([])
  const [imageModels, setImageModels] = React.useState<Model[]>([])
  const [xaiTechniques, setXAITechniques] = React.useState<XAITechnique[]>([])

  // Charger les données au montage du composant
  React.useEffect(() => {
    const loadData = async () => {
      try {
        const [models, techniques] = await Promise.all([
          apiService.getModels(),
          apiService.getXAITechniques(),
        ])
        setAudioModels(models.audio)
        setImageModels(models.image)
        setXAITechniques(techniques)
      } catch (err) {
        console.error('Failed to load initial data:', err)
        setError('Failed to load platform data')
      }
    }
    loadData()
  }, [])

  const compatibleModels = mediaType === 'audio' ? audioModels : mediaType === 'image' ? imageModels : []

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileValidation(file)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      handleFileValidation(file)
    }
  }

  const handleFileValidation = (file: File) => {
    const isAudio = file.type.startsWith('audio/')
    const isImage = file.type.startsWith('image/')

    if (isAudio) {
      setMediaType('audio')
      setSelectedFile(file)
      setSelectedModel(null)
      setError(null)
    } else if (isImage) {
      setMediaType('image')
      setSelectedFile(file)
      setSelectedModel(null)
      setError(null)
    } else {
      setError('File type not supported. Please upload an audio or image file.')
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile || !selectedModel) return

    setStatus('loading')
    setError(null)

    try {
      let response: AnalysisResponse

      if (mediaType === 'audio') {
        response = await apiService.analyzeAudio(
          selectedFile,
          selectedModel,
          selectedXAI
        )
      } else {
        response = await apiService.analyzeImage(
          selectedFile,
          selectedModel,
          selectedXAI
        )
      }

      setResults(response)
      setStatus('success')
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Analysis failed'
      setError(errorMsg)
      setStatus('error')
    }
  }

  const toggleXAI = (technique: string) => {
    setSelectedXAI(prev =>
      prev.includes(technique) ? prev.filter(t => t !== technique) : [...prev, technique]
    )
  }

  const handleNewAnalysis = () => {
    setResults(null)
    setSelectedFile(null)
    setSelectedModel(null)
    setSelectedXAI([])
    setMediaType(null)
    setStatus('idle')
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Upload Section */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-50 mb-2">Upload Media</h2>
          <p className="text-slate-600 dark:text-slate-400 mb-6">
            Upload an audio file for deepfake detection or an X-ray image for lung cancer detection
          </p>

          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className="border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-lg p-8 text-center hover:border-blue-500 dark:hover:border-blue-500 transition-colors cursor-pointer bg-white dark:bg-slate-800"
          >
            <input
              type="file"
              onChange={handleFileSelect}
              accept="audio/*,image/*"
              className="hidden"
              id="file-input"
            />
            <label htmlFor="file-input" className="cursor-pointer">
              <div className="flex flex-col items-center gap-3">
                <Upload className="w-12 h-12 text-slate-400 dark:text-slate-600" />
                <div>
                  <p className="text-lg font-semibold text-slate-900 dark:text-slate-50">
                    Drag and drop your file here
                  </p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    or click to browse (Audio: MP3, WAV | Image: PNG, JPG)
                  </p>
                </div>
              </div>
            </label>
          </div>

          {selectedFile && (
            <div className="mt-4 p-4 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-900 rounded-lg flex items-center gap-3">
              <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0" />
              <div>
                <p className="text-sm text-green-800 dark:text-green-200">
                  File selected: <span className="font-semibold">{selectedFile.name}</span>
                </p>
                <p className="text-xs text-green-700 dark:text-green-300 mt-1">
                  Type: {mediaType === 'audio' ? '🔊 Audio' : '🖼️ Image'}
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-900 rounded-lg flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Model Selection */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm p-6 border border-slate-200 dark:border-slate-700">
              <h3 className="text-xl font-bold text-slate-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-blue-500" />
                Classification Model
              </h3>

              {mediaType === null ? (
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Upload a file to see available models
                </p>
              ) : (
                <div className="space-y-3">
                  {compatibleModels.map(model => (
                    <button
                      key={model.id}
                      onClick={() => setSelectedModel(model.id)}
                      className={`w-full p-3 text-left rounded-lg border-2 transition-all ${
                        selectedModel === model.id
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-950 text-blue-900 dark:text-blue-50'
                          : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-700 text-slate-900 dark:text-slate-50 hover:border-slate-300 dark:hover:border-slate-600'
                      }`}
                    >
                      <p className="font-semibold">{model.name}</p>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* XAI Techniques */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm p-6 border border-slate-200 dark:border-slate-700">
              <h3 className="text-xl font-bold text-slate-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                <Eye className="w-5 h-5 text-purple-500" />
                Explainability Methods
              </h3>

              {selectedModel === null ? (
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Select a model first
                </p>
              ) : (
                <div className="space-y-3">
                  {xaiTechniques.map(technique => (
                    <label
                      key={technique.id}
                      className="flex items-start gap-3 p-3 rounded-lg border-2 border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 cursor-pointer transition-all"
                    >
                      <input
                        type="checkbox"
                        checked={selectedXAI.includes(technique.id)}
                        onChange={() => toggleXAI(technique.id)}
                        className="mt-1 w-5 h-5 cursor-pointer"
                      />
                      <div>
                        <p className="font-semibold text-slate-900 dark:text-slate-50">{technique.name}</p>
                        <p className="text-xs text-slate-600 dark:text-slate-400">{technique.description}</p>
                      </div>
                    </label>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Results */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm p-6 border border-slate-200 dark:border-slate-700 sticky top-6">
              <h3 className="text-xl font-bold text-slate-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-emerald-500" />
                Results
              </h3>

              {!results ? (
                <div className="text-center">
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                    {selectedFile && selectedModel
                      ? 'Ready to analyze'
                      : selectedFile
                        ? 'Select a model to continue'
                        : 'Upload a file to get started'}
                  </p>
                  <button
                    onClick={handleAnalyze}
                    disabled={!selectedFile || !selectedModel || status === 'loading'}
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-shadow"
                  >
                    {status === 'loading' ? 'Analyzing...' : 'Analyze'}
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="p-4 bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-900 rounded-lg">
                    <p className="text-sm font-semibold text-emerald-900 dark:text-emerald-50">
                      Prediction: {results.prediction}
                    </p>
                    <p className="text-sm text-emerald-800 dark:text-emerald-200 mt-1">
                      Confidence: {(results.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  <button
                    onClick={handleNewAnalysis}
                    className="w-full text-blue-600 dark:text-blue-400 font-semibold py-2 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-950 transition-colors"
                  >
                    New Analysis
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* XAI Visualizations */}
        {results && Object.keys(results.xai_results).length > 0 && (
          <div className="mt-12">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-slate-50 mb-6">
              Explainability Results
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(results.xai_results).map(([key, value]: [string, any]) => (
                <div
                  key={key}
                  className="bg-white dark:bg-slate-800 rounded-lg shadow-sm overflow-hidden border border-slate-200 dark:border-slate-700"
                >
                  <div className="bg-gradient-to-r from-blue-500 to-purple-600 px-6 py-4">
                    <h3 className="text-lg font-bold text-white capitalize">{key}</h3>
                  </div>
                  <div className="p-6">
                    {value.error ? (
                      <div className="p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-900 rounded-lg">
                        <p className="text-sm text-red-600 dark:text-red-400">{value.error}</p>
                      </div>
                    ) : value.image ? (
                      <div className="space-y-3">
                        <img
                          src={value.image}
                          alt={`${key} explanation`}
                          className="w-full rounded-lg border border-slate-200 dark:border-slate-700"
                        />
                        <div className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                          {value.type && <p><span className="font-semibold">Method:</span> {value.type}</p>}
                          {value.num_samples && <p><span className="font-semibold">Samples:</span> {value.num_samples}</p>}
                          {value.layer && <p><span className="font-semibold">Layer:</span> {value.layer}</p>}
                          {value.top_label !== undefined && <p><span className="font-semibold">Label:</span> {value.top_label}</p>}
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-900 rounded-lg">
                        <p className="text-sm text-blue-600 dark:text-blue-400">
                          {value.explanation || 'Explanation generated successfully'}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
