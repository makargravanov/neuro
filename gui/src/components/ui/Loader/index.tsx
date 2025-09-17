// src/components/ui/Loader/index.tsx
import { motion, type Variants } from 'framer-motion'; // <-- Импортируем Variants
import styles from './Loader.module.css';

const loaderVariants: Variants = {
    initial: { opacity: 0 },
    animate: {
        opacity: 1,
        transition: { staggerChildren: 0.1 }
    },
    exit: { opacity: 0 }
};

const synapseVariants: Variants = {
    initial: { y: 0 },
    animate: {
        y: [0, -10, 0],
        transition: {
            duration: 0.8,
            repeat: Infinity,
            ease: "easeInOut"
        }
    }
};

export const Loader = () => (
    <motion.div
        className={styles.loaderBackdrop}
        variants={loaderVariants}
        initial="initial"
        animate="animate"
        exit="exit"
    >
        {/* Теперь ошибки не будет */}
        <motion.div className={styles.synapse} variants={synapseVariants} />
        <motion.div className={styles.synapse} variants={synapseVariants} />
        <motion.div className={styles.synapse} variants={synapseVariants} />
    </motion.div>
);