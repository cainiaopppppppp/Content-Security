<template>
  <div>
    <el-table :data="visitorHistory" style="width: 100%">
      <el-table-column prop="student_id" label="学号" width="180"></el-table-column>
      <el-table-column prop="visitor_name" label="姓名" width="180"></el-table-column>
      <el-table-column prop="timestamp" label="签到时间" width="180"></el-table-column>
    </el-table>
  </div>
</template>

<script>
export default {
  data() {
    return {
      visitorHistory: []
    }
  },
  async mounted() {
    try {
      const response = await this.$axios.get('history/')
      this.visitorHistory = response.data
    } catch (error) {
      console.error('Error fetching visitor history:', error)
    }
  }
}
</script>
