// src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import FaceRecognition from '@/components/FaceRecognition.vue'
import VisitorHistroy from '@/components/VisitorHistroy.vue'
import AddVisitor from '@/components/AddVistor.vue'

const routes = [
  { path: '/', name: 'FaceRecognition', component: FaceRecognition },
  { path: '/history', name: 'VisitorHistory', component: VisitorHistroy },
  { path: '/add', name: 'AddVisitor', component: AddVisitor }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

export default router
