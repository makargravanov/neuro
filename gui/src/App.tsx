import styles from './App.module.css';
import { useLocalization } from './context/LocalizationContext';
import { useDatasets } from './hooks/useDatasets';
import { Card } from './components/ui/Card';
import { AvailableDatasets } from './components/AvailableDatasets';
import { LoadedDatasets } from './components/LoadedDatasets';
import { DatasetViewer } from './components/DatasetViewer';
import { Loader } from './components/ui/Loader';

function App() {
    const { t, setLocale, locale } = useLocalization();
    const { available, loaded, viewingData, isLoading, error, handleLoad, handleUnload, handleView } = useDatasets();

    return (
        <div className={styles.appContainer}>
            <header className={styles.header}>
                <div className={styles.langSwitcher}>
                    <button onClick={() => setLocale('en')} disabled={locale === 'en'}>EN</button>
                    <button onClick={() => setLocale('ru')} disabled={locale === 'ru'}>RU</button>
                </div>
            </header>

            {isLoading && <Loader />}
            {error && <div className={styles.error}>{error}</div>}

            <main className={styles.bentoGrid}>
                <div className={styles.gridItem}>
                    <Card title={t('availableDatasets')}>
                        <AvailableDatasets files={available} onLoad={handleLoad} />
                    </Card>
                </div>
                <div className={styles.gridItem}>
                    <Card title={t('loadedDatasets')}>
                        <LoadedDatasets datasets={loaded} onView={handleView} onUnload={handleUnload} />
                    </Card>
                </div>
                <div className={`${styles.gridItem} ${styles.viewer}`}>
                    <Card title="Dataset Viewer">
                        <DatasetViewer data={viewingData} onView={handleView} />
                    </Card>
                </div>
            </main>
        </div>
    );
}

export default App;