// src/context/LocalizationContext.tsx
import { createContext, useContext, useState, type ReactNode } from 'react';
import { locales } from '../localization/locales';

type Locale = 'en' | 'ru';

interface LocalizationContextType {
    locale: Locale;
    setLocale: (locale: Locale) => void;
    t: (key: keyof typeof locales.en) => string;
}

const LocalizationContext = createContext<LocalizationContextType | undefined>(undefined);

export const LocalizationProvider = ({ children }: { children: ReactNode }) => {
    const [locale, setLocale] = useState<Locale>('en');
    const t = (key: keyof typeof locales.en) => locales[locale][key] || key;
    return (
        <LocalizationContext.Provider value={{ locale, setLocale, t }}>
            {children}
        </LocalizationContext.Provider>
    );
};

export const useLocalization = () => {
    const context = useContext(LocalizationContext);
    if (!context) {
        throw new Error('useLocalization must be used within a LocalizationProvider');
    }
    return context;
};