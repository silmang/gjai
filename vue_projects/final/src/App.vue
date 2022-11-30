<template>
  <img alt="Vue logo" src="./assets/logo.png" />
  <HelloWorld msg="Welcome to Your Vue.js App" />

  <!-- https://haenny.tistory.com/249 -->
  <!-- https://vuejsexamples.com/vuejs-components-for-mediarecorder-api/ -->
  <vue-record-audio @result="onResult" />
</template>

<script>
import HelloWorld from "./components/HelloWorld.vue";
import Vue from "vue";
import VueRecord from "@codekraft-studio/vue-record";

// Vue.use(VueRecord);

export default {
  name: "App",
  components: {
    HelloWorld,
  },
  methods: {
    onResult(data) {
      console.log("The blob data:", data);
      console.log("Downloadable audio", window.URL.createObjectURL(data));
    },
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
