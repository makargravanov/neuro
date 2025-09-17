// src/components/ui/Card/index.tsx
import { motion } from 'framer-motion';
import styles from './Card.module.css';
import type {ReactNode} from 'react';

interface CardProps {
    children: ReactNode;
    title: string;
}

export const Card = ({ children, title }: CardProps) => (
    <motion.div
        className={styles.card}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
    >
        <div className={styles.titleWrapper}>
            <h2 className={styles.title}>{title}</h2>
            <div className={styles.glowLine} />
        </div>
        <div className={styles.content}>{children}</div>
    </motion.div>
);