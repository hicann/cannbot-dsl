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
  markdown: {
    math: true
  },
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
            { text: 'Reg矢量计算', link: '/api/kernel/reg_compute/', collapsed: true, items: [
              { text: '概述', link: '/api/kernel/reg_compute/overview' },
              { text: '关键特性说明', link: '/api/kernel/reg_compute/key-features' },
              { text: 'Reg数据搬入', link: '/api/kernel/reg_compute/load/', collapsed: true, items: [
                { text: '概述', link: '/api/kernel/reg_compute/load/overview' },
                { text: 'vload_broadcast', link: '/api/kernel/reg_compute/load/vload-broadcast' },
                { text: 'vload_deinterleave', link: '/api/kernel/reg_compute/load/vload-deinterleave' },
                { text: 'vload_downsample', link: '/api/kernel/reg_compute/load/vload-downsample' },
                { text: 'vload_strided', link: '/api/kernel/reg_compute/load/vload-strided' },
                { text: 'vload_unalign_init', link: '/api/kernel/reg_compute/load/vload-unalign-init' },
                { text: 'vload_unalign', link: '/api/kernel/reg_compute/load/vload-unalign' },
                { text: 'vload_unpack', link: '/api/kernel/reg_compute/load/vload-unpack' },
                { text: 'vload_upsample', link: '/api/kernel/reg_compute/load/vload-upsample' },
                { text: 'vload', link: '/api/kernel/reg_compute/load/vload' },
                { text: 'vmask_load', link: '/api/kernel/reg_compute/load/vmask-load' }
              ] },
              { text: '基础算术', link: '/api/kernel/reg_compute/reg_arith/', collapsed: true, items: [
                { text: 'vabs', link: '/api/kernel/reg_compute/reg_arith/vabs' },
                { text: 'vadd', link: '/api/kernel/reg_compute/reg_arith/vadd' },
                { text: 'vaddc', link: '/api/kernel/reg_compute/reg_arith/vaddc' },
                { text: 'vaddco', link: '/api/kernel/reg_compute/reg_arith/vaddco' },
                { text: 'vadds', link: '/api/kernel/reg_compute/reg_arith/vadds' },
                { text: 'vdiv', link: '/api/kernel/reg_compute/reg_arith/vdiv' },
                { text: 'vexp', link: '/api/kernel/reg_compute/reg_arith/vexp' },
                { text: 'vlog', link: '/api/kernel/reg_compute/reg_arith/vlog' },
                { text: 'vmax', link: '/api/kernel/reg_compute/reg_arith/vmax' },
                { text: 'vmaxs', link: '/api/kernel/reg_compute/reg_arith/vmaxs' },
                { text: 'vmin', link: '/api/kernel/reg_compute/reg_arith/vmin' },
                { text: 'vmins', link: '/api/kernel/reg_compute/reg_arith/vmins' },
                { text: 'vmul', link: '/api/kernel/reg_compute/reg_arith/vmul' },
                { text: 'vmull', link: '/api/kernel/reg_compute/reg_arith/vmull' },
                { text: 'vmuls', link: '/api/kernel/reg_compute/reg_arith/vmuls' },
                { text: 'vneg', link: '/api/kernel/reg_compute/reg_arith/vneg' },
                { text: 'vsqrt', link: '/api/kernel/reg_compute/reg_arith/vsqrt' },
                { text: 'vsub', link: '/api/kernel/reg_compute/reg_arith/vsub' },
                { text: 'vsubbo', link: '/api/kernel/reg_compute/reg_arith/vsubbo' },
                { text: 'vsubc', link: '/api/kernel/reg_compute/reg_arith/vsubc' }
              ] },
              { text: '广播操作', link: '/api/kernel/reg_compute/reg_broadcast/', collapsed: true, items: [
                { text: 'vdup', link: '/api/kernel/reg_compute/reg_broadcast/vdup' },
                { text: 'vdups', link: '/api/kernel/reg_compute/reg_broadcast/vdups' }
              ] },
              { text: '比较计算', link: '/api/kernel/reg_compute/reg_compare/', collapsed: true, items: [
                { text: 'veq', link: '/api/kernel/reg_compute/reg_compare/veq' },
                { text: 'veqs', link: '/api/kernel/reg_compute/reg_compare/veqs' },
                { text: 'vge', link: '/api/kernel/reg_compute/reg_compare/vge' },
                { text: 'vges', link: '/api/kernel/reg_compute/reg_compare/vges' },
                { text: 'vgt', link: '/api/kernel/reg_compute/reg_compare/vgt' },
                { text: 'vgts', link: '/api/kernel/reg_compute/reg_compare/vgts' },
                { text: 'vle', link: '/api/kernel/reg_compute/reg_compare/vle' },
                { text: 'vles', link: '/api/kernel/reg_compute/reg_compare/vles' },
                { text: 'vlt', link: '/api/kernel/reg_compute/reg_compare/vlt' },
                { text: 'vlts', link: '/api/kernel/reg_compute/reg_compare/vlts' },
                { text: 'vne', link: '/api/kernel/reg_compute/reg_compare/vne' },
                { text: 'vnes', link: '/api/kernel/reg_compute/reg_compare/vnes' }
              ] },
              { text: '类型转换', link: '/api/kernel/reg_compute/reg_convert/', collapsed: true, items: [
                { text: 'RoundingMode', link: '/api/kernel/reg_compute/reg_convert/roundingmode' },
                { text: 'vcast', link: '/api/kernel/reg_compute/reg_convert/vcast' },
                { text: 'vceil', link: '/api/kernel/reg_compute/reg_convert/vceil' },
                { text: 'vfloor', link: '/api/kernel/reg_compute/reg_convert/vfloor' },
                { text: 'vreinterpret_lanes', link: '/api/kernel/reg_compute/reg_convert/vreinterpret-lanes' },
                { text: 'vreinterpret', link: '/api/kernel/reg_compute/reg_convert/vreinterpret' },
                { text: 'vtrunc', link: '/api/kernel/reg_compute/reg_convert/vtrunc' }
              ] },
              { text: '复合计算', link: '/api/kernel/reg_compute/reg_fused/', collapsed: true, items: [
                { text: 'vabs_sub', link: '/api/kernel/reg_compute/reg_fused/vabs-sub' },
                { text: 'vaxpy', link: '/api/kernel/reg_compute/reg_fused/vaxpy' },
                { text: 'vcast_exp_sub', link: '/api/kernel/reg_compute/reg_fused/vcast-exp-sub' },
                { text: 'vexp_sub', link: '/api/kernel/reg_compute/reg_fused/vexp-sub' },
                { text: 'vleakyrelu', link: '/api/kernel/reg_compute/reg_fused/vleakyrelu' },
                { text: 'vmadd', link: '/api/kernel/reg_compute/reg_fused/vmadd' },
                { text: 'vmula', link: '/api/kernel/reg_compute/reg_fused/vmula' },
                { text: 'vprelu', link: '/api/kernel/reg_compute/reg_fused/vprelu' },
                { text: 'vrelu', link: '/api/kernel/reg_compute/reg_fused/vrelu' }
              ] },
              { text: '聚合操作', link: '/api/kernel/reg_compute/reg_gather/', collapsed: true, items: [
                { text: 'vgather_reg', link: '/api/kernel/reg_compute/reg_gather/vgather-reg' }
              ] },
              { text: '直方图', link: '/api/kernel/reg_compute/reg_histogram/', collapsed: true, items: [
                { text: 'vhistogram_accumulate', link: '/api/kernel/reg_compute/reg_histogram/vhistogram-accumulate' },
                { text: 'vhistogram_frequency', link: '/api/kernel/reg_compute/reg_histogram/vhistogram-frequency' }
              ] },
              { text: '索引操作', link: '/api/kernel/reg_compute/reg_index/', collapsed: true, items: [
                { text: 'varange', link: '/api/kernel/reg_compute/reg_index/varange' }
              ] },
              { text: '逻辑计算', link: '/api/kernel/reg_compute/reg_logic/', collapsed: true, items: [
                { text: 'mask_and', link: '/api/kernel/reg_compute/reg_logic/mask-and' },
                { text: 'mask_not', link: '/api/kernel/reg_compute/reg_logic/mask-not' },
                { text: 'mask_or', link: '/api/kernel/reg_compute/reg_logic/mask-or' },
                { text: 'mask_xor', link: '/api/kernel/reg_compute/reg_logic/mask-xor' },
                { text: 'vbitwise_and', link: '/api/kernel/reg_compute/reg_logic/vbitwise-and' },
                { text: 'vbitwise_or', link: '/api/kernel/reg_compute/reg_logic/vbitwise-or' },
                { text: 'vbitwise_xor', link: '/api/kernel/reg_compute/reg_logic/vbitwise-xor' },
                { text: 'vmask', link: '/api/kernel/reg_compute/reg_logic/vmask' },
                { text: 'vnot', link: '/api/kernel/reg_compute/reg_logic/vnot' },
                { text: 'vshl', link: '/api/kernel/reg_compute/reg_logic/vshl' },
                { text: 'vshr', link: '/api/kernel/reg_compute/reg_logic/vshr' }
              ] },
              { text: '掩码寄存器操作', link: '/api/kernel/reg_compute/reg_mask/', collapsed: true, items: [
                { text: 'create_mask', link: '/api/kernel/reg_compute/reg_mask/create-mask' },
                { text: 'full_mask', link: '/api/kernel/reg_compute/reg_mask/full-mask' },
                { text: 'mask_counter', link: '/api/kernel/reg_compute/reg_mask/mask-counter' },
                { text: 'update_mask', link: '/api/kernel/reg_compute/reg_mask/update-mask' }
              ] },
              { text: '排布变换', link: '/api/kernel/reg_compute/reg_permute_sel/', collapsed: true, items: [
                { text: 'vcompress', link: '/api/kernel/reg_compute/reg_permute_sel/vcompress' },
                { text: 'vdeinterleave', link: '/api/kernel/reg_compute/reg_permute_sel/vdeinterleave' },
                { text: 'vinterleave', link: '/api/kernel/reg_compute/reg_permute_sel/vinterleave' },
                { text: 'vmerge', link: '/api/kernel/reg_compute/reg_permute_sel/vmerge' },
                { text: 'vpack', link: '/api/kernel/reg_compute/reg_permute_sel/vpack' },
                { text: 'vselect', link: '/api/kernel/reg_compute/reg_permute_sel/vselect' },
                { text: 'vsqueeze_and_storeunalign_finalize', link: '/api/kernel/reg_compute/reg_permute_sel/vsqueeze-and-storeunalign-finalize' },
                { text: 'vsqueeze_and_storeunalign_init', link: '/api/kernel/reg_compute/reg_permute_sel/vsqueeze-and-storeunalign-init' },
                { text: 'vsqueeze_and_storeunalign', link: '/api/kernel/reg_compute/reg_permute_sel/vsqueeze-and-storeunalign' },
                { text: 'vsqueeze', link: '/api/kernel/reg_compute/reg_permute_sel/vsqueeze' },
                { text: 'vstore_unalign_begin', link: '/api/kernel/reg_compute/reg_permute_sel/vstore-unalign-begin' },
                { text: 'vstore_unalign', link: '/api/kernel/reg_compute/reg_permute_sel/vstore-unalign' },
                { text: 'vunpack', link: '/api/kernel/reg_compute/reg_permute_sel/vunpack' }
              ] },
              { text: '归约计算', link: '/api/kernel/reg_compute/reg_reduce/', collapsed: true, items: [
                { text: 'vpair_reduce_sum', link: '/api/kernel/reg_compute/reg_reduce/vpair-reduce-sum' },
                { text: 'vreduce_max_datablock', link: '/api/kernel/reg_compute/reg_reduce/vreduce-max-datablock' },
                { text: 'vreduce_max', link: '/api/kernel/reg_compute/reg_reduce/vreduce-max' },
                { text: 'vreduce_min_datablock', link: '/api/kernel/reg_compute/reg_reduce/vreduce-min-datablock' },
                { text: 'vreduce_min', link: '/api/kernel/reg_compute/reg_reduce/vreduce-min' },
                { text: 'vreduce_sum_datablock', link: '/api/kernel/reg_compute/reg_reduce/vreduce-sum-datablock' },
                { text: 'vreduce_sum', link: '/api/kernel/reg_compute/reg_reduce/vreduce-sum' }
              ] },
              { text: '同步控制', link: '/api/kernel/reg_compute/reg_sync/', collapsed: true, items: [
                { text: 'vmem_bar', link: '/api/kernel/reg_compute/reg_sync/vmem-bar' }
              ] },
              { text: 'Reg离散搬出', link: '/api/kernel/reg_compute/scatter/', collapsed: true, items: [
                { text: 'vscatter', link: '/api/kernel/reg_compute/scatter/vscatter' }
              ] },
              { text: 'Reg数据搬出', link: '/api/kernel/reg_compute/store/', collapsed: true, items: [
                { text: '概述', link: '/api/kernel/reg_compute/store/overview' },
                { text: 'vmask_store', link: '/api/kernel/reg_compute/store/vmask-store' },
                { text: 'vstore_first', link: '/api/kernel/reg_compute/store/vstore-first' },
                { text: 'vstore_interleave', link: '/api/kernel/reg_compute/store/vstore-interleave' },
                { text: 'vstore_mask', link: '/api/kernel/reg_compute/store/vstore-mask' },
                { text: 'vstore_pack', link: '/api/kernel/reg_compute/store/vstore-pack' },
                { text: 'vstore_strided', link: '/api/kernel/reg_compute/store/vstore-strided' },
                { text: 'vstore_unalign_post', link: '/api/kernel/reg_compute/store/vstore-unalign-post' },
                { text: 'vstore', link: '/api/kernel/reg_compute/store/vstore' },
                { text: 'vstorealign_squeeze_status', link: '/api/kernel/reg_compute/store/vstorealign-squeeze-status' }
              ] },
              { text: 'Reg聚合搬入', link: '/api/kernel/reg_compute/ub_gather/', collapsed: true, items: [
                { text: 'vgather_datablock', link: '/api/kernel/reg_compute/ub_gather/vgather-datablock' },
                { text: 'vgather', link: '/api/kernel/reg_compute/ub_gather/vgather' }
              ] }
            ] },
            { text: '标量计算', link: '/api/kernel/scalar_compute/', collapsed: true, items: [
              { text: '位运算', link: '/api/kernel/scalar_compute/scalar_bit/', collapsed: true, items: [
                { text: 'clear_nthbit', link: '/api/kernel/scalar_compute/scalar_bit/clear-nthbit' },
                { text: 'clz', link: '/api/kernel/scalar_compute/scalar_bit/clz' },
                { text: 'ffs', link: '/api/kernel/scalar_compute/scalar_bit/ffs' },
                { text: 'ffz', link: '/api/kernel/scalar_compute/scalar_bit/ffz' },
                { text: 'scalar.popc', link: '/api/kernel/scalar_compute/scalar_bit/scalar-popc' },
                { text: 'set_nthbit', link: '/api/kernel/scalar_compute/scalar_bit/set-nthbit' },
                { text: 'sflbits', link: '/api/kernel/scalar_compute/scalar_bit/sflbits' },
                { text: 'zero_bits_cnt', link: '/api/kernel/scalar_compute/scalar_bit/zero-bits-cnt' }
              ] },
              { text: '类型转换', link: '/api/kernel/scalar_compute/scalar_convert/', collapsed: true, items: [
                { text: 'cast', link: '/api/kernel/scalar_compute/scalar_convert/cast' },
                { text: 'RoundingMode', link: '/api/kernel/scalar_compute/scalar_convert/roundingmode' }
              ] },
              { text: '数据搬入', link: '/api/kernel/scalar_compute/scalar_load/', collapsed: true, items: [
                { text: 'load_bypass', link: '/api/kernel/scalar_compute/scalar_load/load-bypass' }
              ] },
              { text: '数据搬出', link: '/api/kernel/scalar_compute/scalar_store/', collapsed: true, items: [
                { text: 'cube_store_bypass', link: '/api/kernel/scalar_compute/scalar_store/cube-store-bypass' },
                { text: 'vec_store_bypass', link: '/api/kernel/scalar_compute/scalar_store/vec-store-bypass' }
              ] }
            ] },
            { text: '原子操作', link: '/api/kernel/atomic/', collapsed: true, items: [
              { text: '概述', link: '/api/kernel/atomic/overview' },
              { text: '关键特性说明', link: '/api/kernel/atomic/key-features' },
              { text: '标量原子操作', link: '/api/kernel/atomic/scalar_atomic/', collapsed: true, items: [
                { text: 'atomic_add', link: '/api/kernel/atomic/scalar_atomic/atomic-add' },
                { text: 'atomic_and', link: '/api/kernel/atomic/scalar_atomic/atomic-and' },
                { text: 'atomic_cas', link: '/api/kernel/atomic/scalar_atomic/atomic-cas' },
                { text: 'atomic_dec', link: '/api/kernel/atomic/scalar_atomic/atomic-dec' },
                { text: 'atomic_exch', link: '/api/kernel/atomic/scalar_atomic/atomic-exch' },
                { text: 'atomic_inc', link: '/api/kernel/atomic/scalar_atomic/atomic-inc' },
                { text: 'atomic_max', link: '/api/kernel/atomic/scalar_atomic/atomic-max' },
                { text: 'atomic_min', link: '/api/kernel/atomic/scalar_atomic/atomic-min' },
                { text: 'atomic_or', link: '/api/kernel/atomic/scalar_atomic/atomic-or' },
                { text: 'atomic_sub', link: '/api/kernel/atomic/scalar_atomic/atomic-sub' },
                { text: 'atomic_xor', link: '/api/kernel/atomic/scalar_atomic/atomic-xor' }
              ] }
            ] },
            { text: '矩阵计算', link: '/api/kernel/cube_compute/', collapsed: true, items: [
              { text: '概述', link: '/api/kernel/cube_compute/overview' },
              { text: '矩阵计算单元', link: '/api/kernel/cube_compute/unit' },
              { text: '计算流程', link: '/api/kernel/cube_compute/flow' },
              { text: '背景与核心概念', link: '/api/kernel/cube_compute/fractal-intro' },
              { text: '关键分形格式', link: '/api/kernel/cube_compute/fractal-formats' },
              { text: '矩阵计算关键特性说明', link: '/api/kernel/cube_compute/key-features', collapsed: true, items: [
                { text: 'GEMV', link: '/api/kernel/cube_compute/key-features-gemv' },
                { text: 'HF32', link: '/api/kernel/cube_compute/key-features-hf32' },
                { text: 'UnitFlag', link: '/api/kernel/cube_compute/key-features-unit-flag' }
              ] },
              { text: 'enable_fp8', link: '/api/kernel/cube_compute/enable-fp8' },
              { text: 'enable_hf32_trans', link: '/api/kernel/cube_compute/enable-hf32-trans' },
              { text: 'enable_hf32', link: '/api/kernel/cube_compute/enable-hf32' },
              { text: 'enable_hif8', link: '/api/kernel/cube_compute/enable-hif8' },
              { text: 'matmul', link: '/api/kernel/cube_compute/matmul' },
              { text: 'set_fp32_mode', link: '/api/kernel/cube_compute/set-fp32-mode' },
              { text: 'set_hf32_round_mode', link: '/api/kernel/cube_compute/set-hf32-round-mode' },
              { text: 'set_mmad_direction', link: '/api/kernel/cube_compute/set-mmad-direction' }
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
              { text: '概述', link: '/api/kernel/synchronization-cache/overview' },
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
