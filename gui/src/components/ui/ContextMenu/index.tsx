import { motion } from 'framer-motion';
import { useEffect, useRef, type ReactNode } from 'react';
import styles from './ContextMenu.module.css';
import * as React from "react";

interface ContextMenuProps {
    x: number;
    y: number;
    onClose: () => void;
    children: ReactNode;
}

export const ContextMenu = ({ x, y, onClose, children }: ContextMenuProps) => {
    const menuRef = useRef<HTMLDivElement>(null);

    // Закрываем меню при клике вне его
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            // Игнорируем все клики, кроме основного (левая кнопка мыши)
            if (event.button !== 0) return;

            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                onClose();
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [onClose]);

    return (
        <motion.div
            ref={menuRef}
            className={styles.contextMenu}
            style={{ top: y, left: x }}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1, ease: "easeOut" }}
        >
            {children}
        </motion.div>
    );
};

interface MenuItemProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'default' | 'danger';
}

export const MenuItem = ({ children, variant = 'default', className, ...props }: MenuItemProps) => (
    <button className={`${styles.menuItem} ${variant === 'danger' ? styles.danger : ''} ${className}`} {...props}>
        {children}
    </button>
);