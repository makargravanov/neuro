
import {motion, AnimatePresence} from 'framer-motion';
import {useLocalization} from '../../context/LocalizationContext';
import type {PaginatedData} from '../../types';
import {Button} from '../ui/Button';
import styles from './DatasetViewer.module.css';
import {useState} from 'react';
import {ContextMenu, MenuItem} from '../ui/ContextMenu';

interface Props {
    data: PaginatedData | null;
    onView: (id: string, page: number, pageSize: number) => void;
    onRemoveColumn: (datasetId: string, columnName: string) => void;
    onSaveAs: (datasetId: string, newName: string) => void; // Новое свойство
}

interface MenuState {
    x: number;
    y: number;
    columnName: string;
}

export const DatasetViewer = ({data, onView, onRemoveColumn, onSaveAs}: Props) => {
    const {t} = useLocalization();
    const [menu, setMenu] = useState<MenuState | null>(null);
    const [isSaveAsPromptVisible, setIsSaveAsPromptVisible] = useState(false);
    const [newDatasetName, setNewDatasetName] = useState('');

    if (!data) {
        return <p className={styles.placeholder}>{t('noDataSelected')}</p>;
    }

    const handlePrev = () => onView(data.id, data.page - 1, data.pageSize);
    const handleNext = () => onView(data.id, data.page + 1, data.pageSize);

    const handleContextMenu = (event: React.MouseEvent<HTMLTableHeaderCellElement>, columnName: string) => {
        event.preventDefault();
        setMenu({x: event.clientX, y: event.clientY, columnName});
    };

    const handleCloseMenu = () => {
        setMenu(null);
    };

    const handleRemoveClick = () => {
        if (menu) {
            onRemoveColumn(data.id, menu.columnName);
            handleCloseMenu();
        }
    };

    const handleSaveAsClick = () => {
        setIsSaveAsPromptVisible(true);
        setNewDatasetName(`${data.name}_copy`); // Предлагаем имя по умолчанию
    };

    const handleSaveAsConfirm = () => {
        if (newDatasetName.trim() && data) {
            onSaveAs(data.id, newDatasetName.trim());
            setIsSaveAsPromptVisible(false);
            setNewDatasetName('');
        }
    };

    const handleSaveAsCancel = () => {
        setIsSaveAsPromptVisible(false);
        setNewDatasetName('');
    };

    return (
        <motion.div initial={{opacity: 0}} animate={{opacity: 1}} className={styles.viewerContainer}>
            <div className={styles.tableWrapper}>
                <table className={styles.table}>
                    <thead>
                    <tr>
                        {data.headers.map(header => (
                            <th key={header} onContextMenu={(e) => handleContextMenu(e, header)}>
                                {header}
                            </th>
                        ))}
                    </tr>
                    </thead>
                    <AnimatePresence>
                        <tbody>
                        {data.data.map((row, rowIndex) => (
                            <motion.tr
                                key={rowIndex}
                                initial={{opacity: 0}}
                                animate={{opacity: 1}}
                                transition={{delay: rowIndex * 0.02}}
                            >
                                {row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}
                            </motion.tr>
                        ))}
                        </tbody>
                    </AnimatePresence>
                </table>
            </div>
            <div className={styles.pagination}>
                <Button onClick={handlePrev} disabled={data.page <= 1}>&lt; {t('prev')}</Button>
                <span>{t('page')} {data.page} {t('of')} {data.totalPages}</span>
                <Button onClick={handleNext} disabled={data.page >= data.totalPages}>{t('next')} &gt;</Button>
                <Button onClick={handleSaveAsClick} disabled={isSaveAsPromptVisible}>{t('saveAs')}</Button> {/* Кнопка "Сохранить как..." */}
            </div>

            <AnimatePresence>
                {isSaveAsPromptVisible && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        className={styles.saveAsPrompt}
                    >
                        <label htmlFor="newDatasetName">{t('newDatasetNamePrompt')}:</label>
                        <input
                            id="newDatasetName"
                            type="text"
                            value={newDatasetName}
                            onChange={(e) => setNewDatasetName(e.target.value)}
                            placeholder={t('enterNewDatasetName')}
                        />
                        <div className={styles.saveAsPromptButtons}>
                            <Button onClick={handleSaveAsCancel} variant="danger">{t('cancel')}</Button>
                            <Button onClick={handleSaveAsConfirm} disabled={!newDatasetName.trim()}>{t('save')}</Button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
            
            <AnimatePresence>
                {menu && (
                    <ContextMenu x={menu.x} y={menu.y} onClose={handleCloseMenu}>
                        <div className={styles.menuTitle}>Column: <strong>{menu.columnName}</strong></div>
                        <MenuItem variant="danger" onClick={handleRemoveClick}>
                            Remove Column
                        </MenuItem>
                    </ContextMenu>
                )}
            </AnimatePresence>
        </motion.div>
    );
};
