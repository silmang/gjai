<div id="app"> 
    ---- {{ '{{ message }}' }} 
</div>

<script src="https://unpkg.com/vue@3/dist/vue.global.js"></script> 
<script>

const app = Vue.createApp({ 
    data() { 
        return { 
            message: '안녕하세요 Vue!' 
        } 
    } 
})

app.mount('#app')

</script>