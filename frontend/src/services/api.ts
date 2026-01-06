/**
 * API Service for XAI Platform
 * Handles all communication with the backend
 */

const API_BASE_URL = (import.meta as any).env.VITE_API_URL || 'http://localhost:5000/api'

export interface AnalysisResponse {
  model: string
  prediction: string
  confidence: number
  xai_results: Record<string, any>
  probabilities?: Record<string, number>
}

export interface Model {
  id: string
  name: string
  type: string
}

export interface XAITechnique {
  id: string
  name: string
  description: string
}

export interface ModelsResponse {
  audio: Model[]
  image: Model[]
}

export interface XAITechniquesResponse {
  techniques: XAITechnique[]
}

class ApiService {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  /**
   * Check API health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`)
      return response.ok
    } catch (error) {
      console.error('Health check failed:', error)
      return false
    }
  }

  /**
   * Get available models
   */
  async getModels(): Promise<ModelsResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/models`)
      if (!response.ok) throw new Error('Failed to fetch models')
      return await response.json()
    } catch (error) {
      console.error('Error fetching models:', error)
      throw error
    }
  }

  /**
   * Get available XAI techniques
   */
  async getXAITechniques(): Promise<XAITechnique[]> {
    try {
      const response = await fetch(`${this.baseUrl}/xai-techniques`)
      if (!response.ok) throw new Error('Failed to fetch XAI techniques')
      const data = await response.json()
      return data.techniques
    } catch (error) {
      console.error('Error fetching XAI techniques:', error)
      throw error
    }
  }

  /**
   * Analyze audio file for deepfake detection
   */
  async analyzeAudio(
    file: File,
    model: string,
    xaiMethods: string[]
  ): Promise<AnalysisResponse> {
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      xaiMethods.forEach(method => {
        formData.append('xai_methods', method)
      })

      const response = await fetch(`${this.baseUrl}/analyze/audio`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Analysis failed')
      }

      return await response.json()
    } catch (error) {
      console.error('Error analyzing audio:', error)
      throw error
    }
  }

  /**
   * Analyze image file for lung cancer detection
   */
  async analyzeImage(
    file: File,
    model: string,
    xaiMethods: string[],
    modelPath?: string
  ): Promise<AnalysisResponse> {
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      xaiMethods.forEach(method => {
        formData.append('xai_methods', method)
      })
      if (modelPath) {
        formData.append('model_path', modelPath)
      }

      const response = await fetch(`${this.baseUrl}/analyze/image`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Analysis failed')
      }

      return await response.json()
    } catch (error) {
      console.error('Error analyzing image:', error)
      throw error
    }
  }
}

export const apiService = new ApiService()

