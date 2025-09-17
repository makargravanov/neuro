// src/components/ui/Button/index.tsx

import { motion, type HTMLMotionProps } from 'framer-motion';
import type {ReactNode} from 'react';
import styles from './Button.module.css';

interface ButtonProps extends HTMLMotionProps<'button'> {
    children: ReactNode;
    variant?: 'primary' | 'danger';
}

export const Button = ({ children, variant = 'primary', ...props }: ButtonProps) => {
    const buttonClasses = `${styles.button} ${variant === 'danger' ? styles.danger : ''}`;

    return (
        <motion.button
            className={buttonClasses}
            whileHover={{ textShadow: "0 0 8px rgba(255,255,255,0.5)" }}
            transition={{ type: 'spring', stiffness: 400, damping: 17 }}
            {...props}
        >
            {children}
        </motion.button>
    );
};