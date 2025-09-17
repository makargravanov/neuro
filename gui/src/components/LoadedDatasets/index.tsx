import { motion } from 'framer-motion';
import { useLocalization } from '../../context/LocalizationContext';
import type {LoadedDataset} from '../../types';
import { Button } from '../ui/Button';
import styles from './LoadedDatasets.module.css';

interface Props {
    datasets: LoadedDataset[];
    onView: (id: string) => void;
    onUnload: (id: string) => void;
}

// (Можно использовать те же variants, что и в AvailableDatasets)
const listVariants = { /* ... */ };
const itemVariants = { /* ... */ };

export const LoadedDatasets = ({ datasets, onView, onUnload }: Props) => {
    const { t } = useLocalization();

    if (datasets.length === 0) {
        return <p className={styles.placeholder}>{t('noDataLoaded')}</p>;
    }

    return (
        <motion.ul className={styles.list} variants={listVariants} initial="hidden" animate="visible">
            {datasets.map(ds => (
                <motion.li key={ds.id} className={styles.listItem} variants={itemVariants}>
                    <span className={styles.datasetName}>{ds.name}</span>
                    <div className={styles.buttonGroup}>
                        <Button onClick={() => onView(ds.id)}>{t('view')}</Button>
                        <Button onClick={() => onUnload(ds.id)} variant="danger">{t('unload')}</Button>
                    </div>
                </motion.li>
            ))}
        </motion.ul>
    );
};