<template>
  <div id="app">
    <el-container>
      <el-header>
        <el-menu mode="horizontal" :default-active="activeIndex">
          <el-menu-item index="1">
            <router-link to="/">人脸考勤</router-link>
          </el-menu-item>
          <el-menu-item index="2">
            <router-link to="/history">签到记录</router-link>
          </el-menu-item>
          <el-menu-item index="3">
            <router-link to="/add">添加用户</router-link>
          </el-menu-item>
        </el-menu>
      </el-header>

      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </div>
</template>

<script>
export default {
  data() {
    return {
      activeIndex: '1'
    }
  },
  watch: {
    '$route'(to) {
      this.updateActiveIndex(to)
    }
  },
  mounted() {
    this.updateActiveIndex(this.$route)
  },
  methods: {
    updateActiveIndex(route) {
      switch (route.path) {
        case '/':
          this.activeIndex = '1'
          break
        case '/history':
          this.activeIndex = '2'
          break
        case '/add':
          this.activeIndex = '3'
          break
        default:
          this.activeIndex = '1'
      }
    }
  }
}
</script>
