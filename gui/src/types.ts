// src/types/index.ts

export interface LoadedDataset {
    id: string;
    name: string;
}

export interface PaginatedData {
    id: string;
    name: string;
    page: number;
    pageSize: number;
    totalPages: number;
    totalRows: number;
    headers: string[];
    data: string[][];
}