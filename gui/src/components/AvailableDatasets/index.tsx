import { motion } from 'framer-motion';
import { useLocalization } from '../../context/LocalizationContext';
import styles from './AvailableDatasets.module.css';
import { Button } from '../ui/Button';

interface Props {
    files: string[];
    onLoad: (filePath: string) => void;
}

const listVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.05,
        },
    },
};

const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 },
};

export const AvailableDatasets = ({ files, onLoad }: Props) => {
    const { t } = useLocalization();
    return (
        <motion.ul
            className={styles.list}
            variants={listVariants}
            initial="hidden"
            animate="visible"
        >
            {files.map(file => (
                <motion.li key={file} className={styles.listItem} variants={itemVariants}>
                    <span className={styles.fileName}>{file.split('/').pop()}</span>
                    <Button onClick={() => onLoad(file)}>{t('load')}</Button>
                </motion.li>
            ))}
        </motion.ul>
    );
};