<template>
  <div>
    <el-form :model="form" label-width="120px">
      <el-form-item label="姓名">
        <el-input v-model="form.name"></el-input>
      </el-form-item>

      <el-form-item label="学号">
        <el-input v-model="form.student_id"></el-input>
      </el-form-item>

      <el-form-item label="上传自己的图片">
        <input type="file" @change="handleFileChange" accept="image/*">
      </el-form-item>

      <el-button type="success" @click="addVisitor">添加用户</el-button>
    </el-form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      form: {
        name: '',
        student_id: '',
      },
      selectedImage: null,
    };
  },
  methods: {
    handleFileChange(event) {
      const file = event.target.files[0];
      if (file) {
        this.selectedImage = file;
      } else {
        this.selectedImage = null;
      }
    },
    async addVisitor() {
      if (!this.form.name || !this.form.student_id || !this.selectedImage) {
        this.$message.error('信息表单不完整!');
        return;
      }

      try {
        const formData = new FormData();
        formData.append('name', this.form.name);
        formData.append('student_id', this.form.student_id);
        formData.append('face_image', this.selectedImage);
        const response = await this.$axios.post('/visitors/', formData);
        console.log('添加用户:', response.data);
        this.$message.success('成功添加用户!');
        this.form = {
          name: '',
          student_id: '',
        };
        this.selectedImage = null;

      } catch (error) {
        console.error('添加用户失败:', error);
        this.$message.error('添加用户失败');
      }
    },
  },
};
</script>
  