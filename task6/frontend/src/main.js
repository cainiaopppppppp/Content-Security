// src/main.js
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'

// 配置 axios 的全局默认设置
const axiosInstance = axios.create({
    baseURL: 'http://localhost:8000/api', // Replace with your API's base URL
  });

// 创建 Vue 应用并加载插件
const app = createApp(App)
app.use(router)
app.use(ElementPlus)

// 将 axios 设置到全局实例
app.config.globalProperties.$axios = axiosInstance

app.mount('#app')
