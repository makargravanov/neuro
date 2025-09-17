// src/api/datasetAPI.ts
import axios from 'axios';
import type {PaginatedData} from '../types';

const api = axios.create({
    baseURL: '/api/v1',
});

export const getAvailableDatasets = () => api.get<string[]>('/datasets/available');
export const getLoadedDatasets = () => api.get<{ id: string; name: string }[]>('/datasets/loaded');
export const loadDataset = (filePath: string) => api.post<{ datasetId: string }>('/datasets/load', { filePath });
export const unloadDataset = (id: string) => api.delete(`/datasets/${id}`);
export const getDatasetPage = (id: string, page: number, pageSize: number) =>
    api.get<PaginatedData>(`/datasets/${id}`, { params: { page, pageSize } });