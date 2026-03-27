import js from '@eslint/js';
import tsParser from '@typescript-eslint/parser';
import ts from '@typescript-eslint/eslint-plugin';
import react from 'eslint-plugin-react';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import prettier from 'eslint-plugin-prettier';

export default [
    {
        files: ['**/*.{js,jsx,ts,tsx}'],
        languageOptions: {
            ecmaVersion: 'latest',
            sourceType: 'module',
            parser: tsParser,
            parserOptions: { ecmaFeatures: { jsx: true } },
        },
        plugins: {
            '@typescript-eslint': ts,
            react,
            'jsx-a11y': jsxA11y,
            prettier,
        },
        rules: {
            ...js.configs.recommended.rules,
            ...ts.configs.recommended.rules,
            ...react.configs.recommended.rules,
            ...jsxA11y.configs.recommended.rules,

            'prettier/prettier': [
                'error',
                {
                    singleQuote: true,
                    jsxSingleQuote: true,
                    semi: true,
                    trailingComma: 'es5',
                    bracketSpacing: true,
                    arrowParens: 'always',
                    printWidth: 80,
                    tabWidth: 4,
                },
            ],

            quotes: ['error', 'single'],
            'react/self-closing-comp': 'error',
            'react/react-in-jsx-scope': 'off',
        },
        settings: {
            react: { version: 'detect' },
        },
    },
];
