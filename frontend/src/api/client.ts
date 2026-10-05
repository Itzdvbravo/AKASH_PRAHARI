import axios, { AxiosInstance } from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function resolveApiUrl(path: string): string {
  return new URL(path, BASE_URL).toString();
}

export interface ApiError {
  message: string;
  statusCode?: number;
  details?: unknown;
}

export const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});
