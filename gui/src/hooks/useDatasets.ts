// src/hooks/useDatasets.ts
import { useState, useEffect, useCallback } from 'react';
import * as api from '../api/datasetAPI';
import type {LoadedDataset, PaginatedData} from '../types';

export const useDatasets = () => {
    const [available, setAvailable] = useState<string[]>([]);
    const [loaded, setLoaded] = useState<LoadedDataset[]>([]);
    const [viewingData, setViewingData] = useState<PaginatedData | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchAll = useCallback(async () => {
        setIsLoading(true);
        try {
            const [availableRes, loadedRes] = await Promise.all([
                api.getAvailableDatasets(),
                api.getLoadedDatasets(),
            ]);
            setAvailable(availableRes.data);
            setLoaded(loadedRes.data);
        } catch (e) {
            setError('Failed to fetch initial data.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchAll();
    }, [fetchAll]);

    const handleLoad = async (filePath: string) => {
        setIsLoading(true);
        setError(null);
        try {
            await api.loadDataset(filePath);
            await fetchAll(); // Обновляем списки
        } catch (e: any) {
            setError(`Failed to load ${filePath}. ${e.response?.data?.error || e.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleUnload = async (id: string) => {
        setIsLoading(true);
        setError(null);
        try {
            await api.unloadDataset(id);
            if (viewingData?.id === id) setViewingData(null); // Сбрасываем просмотр, если выгрузили текущий
            await fetchAll();
        } catch (e: any) {
            setError(`Failed to unload dataset. ${e.response?.data?.error || e.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleView = async (id: string, page = 1, pageSize = 20) => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await api.getDatasetPage(id, page, pageSize);
            setViewingData(res.data);
        } catch (e: any) {
            setError(`Failed to view dataset. ${e.response?.data?.error || e.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRemoveColumn = async (datasetId: string, columnName: string) => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await api.removeColumn(datasetId, columnName);
            await fetchAll();
            // После удаления колонки, просматриваем новый трансформированный датасет
            await handleView(res.data.newDatasetId);
        } catch (e: any) {
            setError(`Failed to remove column '${columnName}'. ${e.response?.data?.error || e.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSaveAs = async (datasetId: string, newName: string) => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await api.saveDatasetAs(datasetId, newName);
            await fetchAll();
            // Опционально: сразу просмотреть новый сохраненный датасет
            await handleView(res.data.newDatasetId);
        } catch (e: any) {
            setError(`Failed to save dataset as '${newName}'. ${e.response?.data?.error || e.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    return { available, loaded, viewingData, isLoading, error, handleLoad, handleUnload, handleView, handleRemoveColumn, handleSaveAs };
};
