
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
}

interface MenuState {
    x: number;
    y: number;
    columnName: string;
}

export const DatasetViewer = ({data, onView, onRemoveColumn}: Props) => {
    const {t} = useLocalization();
    const [menu, setMenu] = useState<MenuState | null>(null);

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
                <Button onClick={handlePrev} disabled={data.page <= 1}>&lt; Prev</Button>
                <span>Page {data.page} of {data.totalPages}</span>
                <Button onClick={handleNext} disabled={data.page >= data.totalPages}>Next &gt;</Button>
            </div>
            
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