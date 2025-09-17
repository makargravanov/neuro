// src/localization/locales.ts
export const locales = {
    en: {
        header: "Work with CSV",
        availableDatasets: "Available Datasets",
        loadedDatasets: "Loaded Datasets",
        load: "Load",
        view: "View",
        unload: "Unload",
        noDataLoaded: "No dataset loaded in memory.",
        noDataSelected: "Select a loaded dataset to view its contents.",
    },
    ru: {
        header: "Работа с CSV",
        availableDatasets: "Доступные датасеты",
        loadedDatasets: "Загруженные датасеты",
        load: "Загрузить",
        view: "Смотреть",
        unload: "Выгрузить",
        noDataLoaded: "Нет загруженных датасетов в памяти.",
        noDataSelected: "Выберите загруженный датасет для просмотра.",
    },
};

export type LocaleKey = keyof typeof locales.ru;