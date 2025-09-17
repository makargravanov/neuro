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
        try {
            await api.loadDataset(filePath);
            await fetchAll(); // Обновляем списки
        } catch (e) {
            setError(`Failed to load ${filePath}.`);
        }
    };

    const handleUnload = async (id: string) => {
        try {
            await api.unloadDataset(id);
            if (viewingData?.id === id) setViewingData(null); // Сбрасываем просмотр, если выгрузили текущий
            await fetchAll();
        } catch (e) {
            setError(`Failed to unload dataset.`);
        }
    };

    const handleView = async (id: string, page = 1, pageSize = 20) => {
        setIsLoading(true);
        try {
            const res = await api.getDatasetPage(id, page, pageSize);
            setViewingData(res.data);
        } catch (e) {
            setError(`Failed to view dataset.`);
        } finally {
            setIsLoading(false);
        }
    };

    return { available, loaded, viewingData, isLoading, error, handleLoad, handleUnload, handleView };
};