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

    // Автокоррекция координат, чтобы меню не выходило за пределы окна
    const [menuPos, setMenuPos] = React.useState({ x, y });
    const menuWidth = 200; // примерная ширина меню (px)
    const menuHeight = 80; // примерная высота меню (px)

    React.useEffect(() => {
        let newX = x;
        let newY = y;
        if (typeof window !== 'undefined') {
            if (x + menuWidth > window.innerWidth) {
                newX = window.innerWidth - menuWidth - 10;
            }
            if (y + menuHeight > window.innerHeight) {
                newY = window.innerHeight - menuHeight - 10;
            }
        }
        setMenuPos({ x: newX, y: newY });
    }, [x, y]);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (event.button !== 0) return;
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                console.log('[Debug] ContextMenu: Закрытие по клику вне меню', event);
                onClose();
            }
        };
        const timeout = setTimeout(() => {
            document.addEventListener('mousedown', handleClickOutside);
        }, 0);
        return () => {
            clearTimeout(timeout);
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [onClose]);

    console.log('[Debug] ContextMenu menuPos', menuPos);
    return (
        <div
            ref={menuRef}
            className={styles.contextMenu}
            style={{
                top: 100,
                left: 100,
                position: 'fixed'
            }}
        >
            {children}
        </div>
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