import { defineConfig } from 'vitepress'

const guideSidebar = [
  { text: '文档概览', link: '/getting-started/' },
  { text: '项目介绍', link: '/guide/' },
  { text: '仓库结构', link: '/guide/repository' },
  { text: '样例导航', link: '/examples/' }
]

export default defineConfig({
  lang: 'zh-CN',
  title: 'CANNBot-DSL',
  description: '面向 Ascend NPU 的算子开发项目文档',
  base: process.env.DOCS_BASE || '/',
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['meta', { name: 'theme-color', content: '#2f5bea' }],
    ['link', { rel: 'icon', href: '/logo.svg', type: 'image/svg+xml' }]
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'CANNBot-DSL',
    nav: [
      { text: '开始使用', link: '/getting-started/' },
      { text: '样例', link: '/examples/' },
      { text: 'API 文档', link: '/api/' },
      { text: '参与贡献', link: '/community/contributing' },
      { text: '关于', items: [
        { text: '文档范围', link: '/about/scope' },
        { text: '发布与部署', link: '/about/deployment' }
      ] }
    ],
    sidebar: {
      '/getting-started/': [{ text: '开始使用', items: guideSidebar }],
      '/guide/': [{ text: '项目指南', items: guideSidebar }],
      '/examples/': [{ text: '样例', items: [
        { text: '样例导航', link: '/examples/' },
        { text: '运行测试', link: '/examples/testing' }
      ] }],
      '/api/': [{
        text: 'API 文档',
        link: '/api/',
        items: [
          { text: 'Host API', link: '/api/host/', collapsed: false, items: [
            { text: '平台信息', link: '/api/host/platform-profiling/', collapsed: false, items: [
              { text: 'get_platform_info', link: '/api/host/platform-profiling/get-platform-info' },
              { text: 'get_mem_size', link: '/api/host/platform-profiling/get-mem-size' }
            ] }
          ] },
          { text: 'Kernel API', link: '/api/kernel/', collapsed: false, items: [
            { text: '数据搬运', link: '/api/kernel/data-movement/', collapsed: false, items: [
              { text: 'mem_copy', link: '/api/kernel/data-movement/mem-copy' }
            ] },
            { text: '系统变量访问', link: '/api/kernel/system/', collapsed: false, items: [
              { text: 'get_block_idx', link: '/api/kernel/system/get-block-idx' },
              { text: 'get_block_num', link: '/api/kernel/system/get-block-num' },
              { text: 'get_subblock_id', link: '/api/kernel/system/get-subblock-id' },
              { text: 'get_subblock_dim', link: '/api/kernel/system/get-subblock-dim' },
              { text: 'get_core_id', link: '/api/kernel/system/get-core-id' },
              { text: 'get_system_cycle', link: '/api/kernel/system/get-system-cycle' },
              { text: 'get_status', link: '/api/kernel/system/get-status' },
              { text: 'get_vf_len', link: '/api/kernel/system/get-vf-len' },
              { text: 'get_squeeze_status', link: '/api/kernel/system/get-squeeze-status' }
            ] },
            { text: '同步与缓存控制', link: '/api/kernel/synchronization-cache/', collapsed: false, items: [
              { text: 'dcci_single', link: '/api/kernel/synchronization-cache/dcci-single' },
              { text: 'dcci_entire_out', link: '/api/kernel/synchronization-cache/dcci-entire-out' },
              { text: 'dcci_entire_atomic', link: '/api/kernel/synchronization-cache/dcci-entire-atomic' },
              { text: 'dci', link: '/api/kernel/synchronization-cache/dci' }
            ] }
          ] },
          { text: 'AI CPU API', link: '/api/aicpu/' },
          { text: '通用接口文档模板', link: '/api/api-template' },
        ]
      }],
      '/community/': [{ text: '社区', items: [
        { text: '参与贡献', link: '/community/contributing' }
      ] }],
      '/about/': [{ text: '关于', items: [
        { text: '文档范围', link: '/about/scope' },
        { text: '发布与部署', link: '/about/deployment' }
      ] }]
    },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '没有找到相关内容',
            resetButtonTitle: '清空搜索',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一页', next: '下一页' },
    sidebarMenuLabel: '目录',
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '切换主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    socialLinks: [
      { icon: 'github', link: 'https://gitcode.com/cann/cannbot-dsl' }
    ],
    footer: {
      message: 'CANNBot-DSL 开源项目文档',
      copyright: '基于 CANN Open Software License Agreement Version 2.0'
    }
  },
  sitemap: {
    hostname: 'https://cannbot-dsl.gitcode.com'
  }
})
