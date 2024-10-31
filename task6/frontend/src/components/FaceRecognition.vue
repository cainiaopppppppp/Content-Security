<template>
  <div>
    <video ref="video" width="640" height="480" autoplay></video>
    <canvas ref="canvas" width="640" height="480" style="display: none;"></canvas>
    <div v-if="capturedImage">
      <img :src="capturedImage" alt="Captured Image" width="640" height="480">
    </div>

    <!-- 模型选择和操作按钮 -->
    <div>
      <el-input v-model="studentId" placeholder="输入学号"></el-input>
      <el-select v-model="selectedModel" placeholder="选择模型">
        <el-option label="Model 1 (Default)" value="moedl1"></el-option>
        <el-option label="Model 2 (Alternative)" value="model2"></el-option>
        <el-option label="Model 3 (Alternative)" value="model3"></el-option>
        <!-- 可以添加更多模型 -->
      </el-select>
      <el-button @click="captureImage">截取实时图片</el-button>
      <el-button @click="uploadImage" type="primary" v-if="capturedImage">上传图片</el-button>
      <el-button @click="resetImage" v-if="capturedImage">重新获取图片</el-button>
    </div>

    <!-- 显示结果或消息 -->
    <div v-if="message">
      <el-alert :title="message" type="success" v-if="isSuccess" show-icon></el-alert>
      <el-alert :title="message" type="error" v-else show-icon></el-alert>
    </div>

    <!-- 显示已验证的访客信息 -->
    <div v-if="visitorInfo">
      <h3>Visitor Information:</h3>
      <p><strong>学号:</strong> {{ visitorInfo.student_id }}</p>
      <p><strong>姓名:</strong> {{ visitorInfo.name }}</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      videoStream: null,
      capturedImage: null,
      studentId: '',
      selectedModel: 'model1',
      message: '',
      isSuccess: false,
      visitorInfo: null
    };
  },
  mounted() {
    this.startCamera();
  },
  beforeDestroy() {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  },
  methods: {
    async startCamera() {
      try {
        this.videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        this.$refs.video.srcObject = this.videoStream;
      } catch (error) {
        console.error('无法获取摄像头权限:', error);
      }
    },
    captureImage() {
      const video = this.$refs.video;
      const canvas = this.$refs.canvas;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      this.capturedImage = canvas.toDataURL('image/png');
    },
    async uploadImage() {
      if (!this.studentId) {
        this.$message.error('请输入学号!');
        return;
      }
      console.log(this.selectedModel);

      try {
        const base64Data = this.capturedImage.split(',')[1];
        const response = await this.$axios.post('/validate-visitor/', {
          image: base64Data,
          student_id: this.studentId,
          model: this.selectedModel
        });

        this.visitorInfo = response.data.visitor;
        this.message = response.data.result;
        this.isSuccess = true;
      } catch (error) {
        console.error('上传图片错误:', error);
        this.message = '签到失败';
        this.isSuccess = false;
      }
    },
    resetImage() {
      this.capturedImage = null;
      this.message = '';
      this.visitorInfo = null;
    }
  }
};
</script>
