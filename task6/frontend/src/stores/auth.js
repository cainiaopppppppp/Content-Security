// src/stores/auth.js
import { defineStore } from 'pinia';

export const useAuthStore = defineStore('auth', {
    state: () => ({
        isAuthenticated: false,
        token: null
    }),
    getters: {
        getToken: (state) => state.token,
        getIsAuthenticated: (state) => state.isAuthenticated
    },
    actions: {
        setToken(token) {
            this.token = token;
            this.isAuthenticated = !!token;
            localStorage.setItem('token', token);  // Optionally save token to local storage
        },
        clearToken() {
            this.token = null;
            this.isAuthenticated = false;
            localStorage.removeItem('token');  // Optionally clear token from local storage
        }
    }
});
