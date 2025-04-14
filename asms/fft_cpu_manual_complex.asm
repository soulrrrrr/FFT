
./fft_cpu:     file format elf64-x86-64


Disassembly of section .init:

0000000000002000 <_init>:
    2000:	48 83 ec 08          	sub    $0x8,%rsp
    2004:	48 8b 05 cd 5f 00 00 	mov    0x5fcd(%rip),%rax        # 7fd8 <__gmon_start__@Base>
    200b:	48 85 c0             	test   %rax,%rax
    200e:	74 02                	je     2012 <_init+0x12>
    2010:	ff d0                	call   *%rax
    2012:	48 83 c4 08          	add    $0x8,%rsp
    2016:	c3                   	ret

Disassembly of section .plt:

0000000000002020 <_ZNSo3putEc@plt-0x10>:
    2020:	ff 35 ca 5f 00 00    	push   0x5fca(%rip)        # 7ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
    2026:	ff 25 cc 5f 00 00    	jmp    *0x5fcc(%rip)        # 7ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
    202c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002030 <_ZNSo3putEc@plt>:
    2030:	ff 25 ca 5f 00 00    	jmp    *0x5fca(%rip)        # 8000 <_ZNSo3putEc@GLIBCXX_3.4>
    2036:	68 00 00 00 00       	push   $0x0
    203b:	e9 e0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>:
    2040:	ff 25 c2 5f 00 00    	jmp    *0x5fc2(%rip)        # 8008 <_ZNSt6chrono3_V212system_clock3nowEv@GLIBCXX_3.4.19>
    2046:	68 01 00 00 00       	push   $0x1
    204b:	e9 d0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>:
    2050:	ff 25 ba 5f 00 00    	jmp    *0x5fba(%rip)        # 8010 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@GLIBCXX_3.4>
    2056:	68 02 00 00 00       	push   $0x2
    205b:	e9 c0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002060 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@plt>:
    2060:	ff 25 b2 5f 00 00    	jmp    *0x5fb2(%rip)        # 8018 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@GLIBCXX_3.4.21>
    2066:	68 03 00 00 00       	push   $0x3
    206b:	e9 b0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002070 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@plt>:
    2070:	ff 25 aa 5f 00 00    	jmp    *0x5faa(%rip)        # 8020 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    2076:	68 04 00 00 00       	push   $0x4
    207b:	e9 a0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002080 <_ZNSt8ios_baseC2Ev@plt>:
    2080:	ff 25 a2 5f 00 00    	jmp    *0x5fa2(%rip)        # 8028 <_ZNSt8ios_baseC2Ev@GLIBCXX_3.4>
    2086:	68 05 00 00 00       	push   $0x5
    208b:	e9 90 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002090 <_ZNSt8ios_baseD2Ev@plt>:
    2090:	ff 25 9a 5f 00 00    	jmp    *0x5f9a(%rip)        # 8030 <_ZNSt8ios_baseD2Ev@GLIBCXX_3.4>
    2096:	68 06 00 00 00       	push   $0x6
    209b:	e9 80 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020a0 <__cxa_begin_catch@plt>:
    20a0:	ff 25 92 5f 00 00    	jmp    *0x5f92(%rip)        # 8038 <__cxa_begin_catch@CXXABI_1.3>
    20a6:	68 07 00 00 00       	push   $0x7
    20ab:	e9 70 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020b0 <strlen@plt>:
    20b0:	ff 25 8a 5f 00 00    	jmp    *0x5f8a(%rip)        # 8040 <strlen@GLIBC_2.2.5>
    20b6:	68 08 00 00 00       	push   $0x8
    20bb:	e9 60 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020c0 <_ZSt20__throw_length_errorPKc@plt>:
    20c0:	ff 25 82 5f 00 00    	jmp    *0x5f82(%rip)        # 8048 <_ZSt20__throw_length_errorPKc@GLIBCXX_3.4>
    20c6:	68 09 00 00 00       	push   $0x9
    20cb:	e9 50 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>:
    20d0:	ff 25 7a 5f 00 00    	jmp    *0x5f7a(%rip)        # 8050 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@GLIBCXX_3.4.21>
    20d6:	68 0a 00 00 00       	push   $0xa
    20db:	e9 40 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020e0 <_ZNSo5flushEv@plt>:
    20e0:	ff 25 72 5f 00 00    	jmp    *0x5f72(%rip)        # 8058 <_ZNSo5flushEv@GLIBCXX_3.4>
    20e6:	68 0b 00 00 00       	push   $0xb
    20eb:	e9 30 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020f0 <_ZSt19__throw_logic_errorPKc@plt>:
    20f0:	ff 25 6a 5f 00 00    	jmp    *0x5f6a(%rip)        # 8060 <_ZSt19__throw_logic_errorPKc@GLIBCXX_3.4>
    20f6:	68 0c 00 00 00       	push   $0xc
    20fb:	e9 20 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002100 <cosf@plt>:
    2100:	ff 25 62 5f 00 00    	jmp    *0x5f62(%rip)        # 8068 <cosf@GLIBC_2.2.5>
    2106:	68 0d 00 00 00       	push   $0xd
    210b:	e9 10 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002110 <memcpy@plt>:
    2110:	ff 25 5a 5f 00 00    	jmp    *0x5f5a(%rip)        # 8070 <memcpy@GLIBC_2.14>
    2116:	68 0e 00 00 00       	push   $0xe
    211b:	e9 00 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002120 <sinf@plt>:
    2120:	ff 25 52 5f 00 00    	jmp    *0x5f52(%rip)        # 8078 <sinf@GLIBC_2.2.5>
    2126:	68 0f 00 00 00       	push   $0xf
    212b:	e9 f0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>:
    2130:	ff 25 4a 5f 00 00    	jmp    *0x5f4a(%rip)        # 8080 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@GLIBCXX_3.4>
    2136:	68 10 00 00 00       	push   $0x10
    213b:	e9 e0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002140 <_Znwm@plt>:
    2140:	ff 25 42 5f 00 00    	jmp    *0x5f42(%rip)        # 8088 <_Znwm@GLIBCXX_3.4>
    2146:	68 11 00 00 00       	push   $0x11
    214b:	e9 d0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002150 <_ZdlPvm@plt>:
    2150:	ff 25 3a 5f 00 00    	jmp    *0x5f3a(%rip)        # 8090 <_ZdlPvm@CXXABI_1.3.9>
    2156:	68 12 00 00 00       	push   $0x12
    215b:	e9 c0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>:
    2160:	ff 25 32 5f 00 00    	jmp    *0x5f32(%rip)        # 8098 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@GLIBCXX_3.4>
    2166:	68 13 00 00 00       	push   $0x13
    216b:	e9 b0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
    2170:	ff 25 2a 5f 00 00    	jmp    *0x5f2a(%rip)        # 80a0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
    2176:	68 14 00 00 00       	push   $0x14
    217b:	e9 a0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>:
    2180:	ff 25 22 5f 00 00    	jmp    *0x5f22(%rip)        # 80a8 <_ZNKSt5ctypeIcE13_M_widen_initEv@GLIBCXX_3.4.11>
    2186:	68 15 00 00 00       	push   $0x15
    218b:	e9 90 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>:
    2190:	ff 25 1a 5f 00 00    	jmp    *0x5f1a(%rip)        # 80b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@GLIBCXX_3.4.21>
    2196:	68 16 00 00 00       	push   $0x16
    219b:	e9 80 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021a0 <_ZSt16__throw_bad_castv@plt>:
    21a0:	ff 25 12 5f 00 00    	jmp    *0x5f12(%rip)        # 80b8 <_ZSt16__throw_bad_castv@GLIBCXX_3.4>
    21a6:	68 17 00 00 00       	push   $0x17
    21ab:	e9 70 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>:
    21b0:	ff 25 0a 5f 00 00    	jmp    *0x5f0a(%rip)        # 80c0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@GLIBCXX_3.4>
    21b6:	68 18 00 00 00       	push   $0x18
    21bb:	e9 60 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021c0 <_ZNSt6localeD1Ev@plt>:
    21c0:	ff 25 02 5f 00 00    	jmp    *0x5f02(%rip)        # 80c8 <_ZNSt6localeD1Ev@GLIBCXX_3.4>
    21c6:	68 19 00 00 00       	push   $0x19
    21cb:	e9 50 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>:
    21d0:	ff 25 fa 5e 00 00    	jmp    *0x5efa(%rip)        # 80d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    21d6:	68 1a 00 00 00       	push   $0x1a
    21db:	e9 40 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>:
    21e0:	ff 25 f2 5e 00 00    	jmp    *0x5ef2(%rip)        # 80d8 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    21e6:	68 1b 00 00 00       	push   $0x1b
    21eb:	e9 30 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@plt>:
    21f0:	ff 25 ea 5e 00 00    	jmp    *0x5eea(%rip)        # 80e0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@GLIBCXX_3.4.21>
    21f6:	68 1c 00 00 00       	push   $0x1c
    21fb:	e9 20 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002200 <_ZNSo9_M_insertIdEERSoT_@plt>:
    2200:	ff 25 e2 5e 00 00    	jmp    *0x5ee2(%rip)        # 80e8 <_ZNSo9_M_insertIdEERSoT_@GLIBCXX_3.4.9>
    2206:	68 1d 00 00 00       	push   $0x1d
    220b:	e9 10 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002210 <memmove@plt>:
    2210:	ff 25 da 5e 00 00    	jmp    *0x5eda(%rip)        # 80f0 <memmove@GLIBC_2.2.5>
    2216:	68 1e 00 00 00       	push   $0x1e
    221b:	e9 00 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002220 <__cxa_end_catch@plt>:
    2220:	ff 25 d2 5e 00 00    	jmp    *0x5ed2(%rip)        # 80f8 <__cxa_end_catch@CXXABI_1.3>
    2226:	68 1f 00 00 00       	push   $0x1f
    222b:	e9 f0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>:
    2230:	ff 25 ca 5e 00 00    	jmp    *0x5eca(%rip)        # 8100 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@GLIBCXX_3.4>
    2236:	68 20 00 00 00       	push   $0x20
    223b:	e9 e0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002240 <_Unwind_Resume@plt>:
    2240:	ff 25 c2 5e 00 00    	jmp    *0x5ec2(%rip)        # 8108 <_Unwind_Resume@GCC_3.0>
    2246:	68 21 00 00 00       	push   $0x21
    224b:	e9 d0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002250 <log2@plt>:
    2250:	ff 25 ba 5e 00 00    	jmp    *0x5eba(%rip)        # 8110 <log2@GLIBC_2.29>
    2256:	68 22 00 00 00       	push   $0x22
    225b:	e9 c0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002260 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>:
    2260:	ff 25 b2 5e 00 00    	jmp    *0x5eb2(%rip)        # 8118 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@GLIBCXX_3.4.21>
    2266:	68 23 00 00 00       	push   $0x23
    226b:	e9 b0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002270 <_ZNSt12__basic_fileIcED1Ev@plt>:
    2270:	ff 25 aa 5e 00 00    	jmp    *0x5eaa(%rip)        # 8120 <_ZNSt12__basic_fileIcED1Ev@GLIBCXX_3.4>
    2276:	68 24 00 00 00       	push   $0x24
    227b:	e9 a0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002280 <__mulsc3@plt>:
    2280:	ff 25 a2 5e 00 00    	jmp    *0x5ea2(%rip)        # 8128 <__mulsc3@GCC_4.0.0>
    2286:	68 25 00 00 00       	push   $0x25
    228b:	e9 90 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002290 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>:
    2290:	ff 25 9a 5e 00 00    	jmp    *0x5e9a(%rip)        # 8130 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@GLIBCXX_3.4.21>
    2296:	68 26 00 00 00       	push   $0x26
    229b:	e9 80 fd ff ff       	jmp    2020 <_init+0x20>

00000000000022a0 <_ZNSi10_M_extractIfEERSiRT_@plt>:
    22a0:	ff 25 92 5e 00 00    	jmp    *0x5e92(%rip)        # 8138 <_ZNSi10_M_extractIfEERSiRT_@GLIBCXX_3.4.9>
    22a6:	68 27 00 00 00       	push   $0x27
    22ab:	e9 70 fd ff ff       	jmp    2020 <_init+0x20>

Disassembly of section .plt.got:

00000000000022b0 <__cxa_finalize@plt>:
    22b0:	ff 25 0a 5d 00 00    	jmp    *0x5d0a(%rip)        # 7fc0 <__cxa_finalize@GLIBC_2.2.5>
    22b6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000000022c0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0.cold>:
  template<typename _Facet>
    inline const _Facet&
    __check_facet(const _Facet* __f)
    {
      if (!__f)
	__throw_bad_cast();
    22c0:	e8 db fe ff ff       	call   21a0 <_ZSt16__throw_bad_castv@plt>

00000000000022c5 <main.cold>:
      basic_string(const _CharT* __s, const _Alloc& __a = _Alloc())
      : _M_dataplus(_M_local_data(), __a)
      {
	// NB: Not required, but considered best practice.
	if (__s == 0)
	  std::__throw_logic_error(__N("basic_string: "
    22c5:	48 8d 3d a4 3d 00 00 	lea    0x3da4(%rip),%rdi        # 6070 <_IO_stdin_used+0x70>
    22cc:	e8 1f fe ff ff       	call   20f0 <_ZSt19__throw_logic_errorPKc@plt>

      _GLIBCXX20_CONSTEXPR
      ~_Vector_base() _GLIBCXX_NOEXCEPT
      {
	_M_deallocate(_M_impl._M_start,
		      _M_impl._M_end_of_storage - _M_impl._M_start);
    22d1:	49 89 c4             	mov    %rax,%r12
    22d4:	c5 f8 77             	vzeroupper
    22d7:	48 8b b5 08 fd ff ff 	mov    -0x2f8(%rbp),%rsi
    22de:	48 8b 85 30 fd ff ff 	mov    -0x2d0(%rbp),%rax
    22e5:	48 29 c6             	sub    %rax,%rsi
      _GLIBCXX20_CONSTEXPR
      void
      _M_deallocate(pointer __p, size_t __n)
      {
	typedef __gnu_cxx::__alloc_traits<_Tp_alloc_type> _Tr;
	if (__p)
    22e8:	48 85 c0             	test   %rax,%rax
    22eb:	74 08                	je     22f5 <main.cold+0x30>
	    _GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n),
				     std::align_val_t(alignof(_Tp)));
	    return;
	  }
#endif
	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
    22ed:	48 89 c7             	mov    %rax,%rdi
    22f0:	e8 5b fe ff ff       	call   2150 <_ZdlPvm@plt>
      ~__new_allocator() _GLIBCXX_USE_NOEXCEPT { }
    22f5:	4c 89 e7             	mov    %r12,%rdi
    22f8:	e8 43 ff ff ff       	call   2240 <_Unwind_Resume@plt>
      /**
       *  @brief  Destroy the string instance.
       */
      _GLIBCXX20_CONSTEXPR
      ~basic_string()
      { _M_dispose(); }
    22fd:	48 89 df             	mov    %rbx,%rdi
    2300:	c5 f8 77             	vzeroupper
    2303:	e8 88 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2308:	48 83 bd 40 fd ff ff 	cmpq   $0x0,-0x2c0(%rbp)
    230f:	00 
    2310:	74 c5                	je     22d7 <main.cold+0x12>
	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
    2312:	48 8b b5 28 fd ff ff 	mov    -0x2d8(%rbp),%rsi
    2319:	48 8b bd 40 fd ff ff 	mov    -0x2c0(%rbp),%rdi
    2320:	e8 2b fe ff ff       	call   2150 <_ZdlPvm@plt>
       *  Calls <tt> a.deallocate(p, n) </tt>
      */
      [[__gnu__::__always_inline__]]
      static _GLIBCXX20_CONSTEXPR void
      deallocate(allocator_type& __a, pointer __p, size_type __n)
      { __a.deallocate(__p, __n); }
    2325:	eb b0                	jmp    22d7 <main.cold+0x12>
    std::ofstream fout(path);
    for (const auto &c : data)
    {
        fout << c.real() << " " << c.imag() << "\n";
    }
    2327:	4c 89 e7             	mov    %r12,%rdi
    232a:	c5 f8 77             	vzeroupper
    232d:	49 89 dc             	mov    %rbx,%r12
    2330:	e8 9b fe ff ff       	call   21d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    2335:	48 8b bd 20 fd ff ff 	mov    -0x2e0(%rbp),%rdi
    233c:	e8 4f fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
      ~__new_allocator() _GLIBCXX_USE_NOEXCEPT { }
    2341:	eb c5                	jmp    2308 <main.cold+0x43>
      virtual
      ~basic_filebuf()
      {
	__try
	  { this->close(); }
	__catch(...)
    2343:	c5 f8 77             	vzeroupper
    2346:	e8 55 fd ff ff       	call   20a0 <__cxa_begin_catch@plt>
    234b:	e8 d0 fe ff ff       	call   2220 <__cxa_end_catch@plt>
    2350:	e9 73 06 00 00       	jmp    29c8 <main+0x5e8>
      // Called by constructors to check initial size.
      static _GLIBCXX20_CONSTEXPR size_type
      _S_check_init_len(size_type __n, const allocator_type& __a)
      {
	if (__n > _S_max_size(_Tp_alloc_type(__a)))
	  __throw_length_error(
    2355:	48 8d 3d 4c 3d 00 00 	lea    0x3d4c(%rip),%rdi        # 60a8 <_IO_stdin_used+0xa8>
    235c:	e8 5f fd ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
    2361:	4c 89 e7             	mov    %r12,%rdi
    2364:	c5 f8 77             	vzeroupper
    2367:	e8 24 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    236c:	48 89 df             	mov    %rbx,%rdi
    236f:	e8 cc fe ff ff       	call   2240 <_Unwind_Resume@plt>
		     ios_base::openmode __mode = ios_base::out)
      : __ostream_type(), _M_filebuf()
      {
	this->init(&_M_filebuf);
	this->open(__s, __mode);
      }
    2374:	48 8b bd 18 fd ff ff 	mov    -0x2e8(%rbp),%rdi
    237b:	c5 f8 77             	vzeroupper
    237e:	e8 5d fe ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
       *  @brief  Base destructor.
       *
       *  This does very little apart from providing a virtual base dtor.
      */
      virtual
      ~basic_ostream() { }
    2383:	48 8b 05 c6 58 00 00 	mov    0x58c6(%rip),%rax        # 7c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    238a:	48 8b 0d c7 58 00 00 	mov    0x58c7(%rip),%rcx        # 7c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2391:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    2398:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    239c:	48 89 8c 05 d0 fd ff 	mov    %rcx,-0x230(%rbp,%rax,1)
    23a3:	ff 
       *
       *  The destructor does nothing.  More specifically, it does not
       *  destroy the streambuf held by rdbuf().
      */
      virtual
      ~basic_ios() { }
    23a4:	48 8b bd 10 fd ff ff 	mov    -0x2f0(%rbp),%rdi
    23ab:	48 8d 05 06 58 00 00 	lea    0x5806(%rip),%rax        # 7bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    23b2:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    23b9:	e8 d2 fc ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    23be:	e9 72 ff ff ff       	jmp    2335 <main.cold+0x70>
    23c3:	4c 89 e7             	mov    %r12,%rdi
    23c6:	c5 f8 77             	vzeroupper
    23c9:	4d 89 ec             	mov    %r13,%r12
    23cc:	e8 bf fd ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    23d1:	48 89 df             	mov    %rbx,%rdi
    23d4:	e8 b7 fd ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    23d9:	e9 2a ff ff ff       	jmp    2308 <main.cold+0x43>
    23de:	66 90                	xchg   %ax,%ax

00000000000023e0 <main>:
    fftr(N, 1, 0, (complex_t *)data.data(), (complex_t *)out.data());
    std::copy(data.begin(), data.end(), out.begin()); // for fft stockham
}

int main(int argc, char *argv[])
{
    23e0:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    23e5:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    23e9:	41 ff 72 f8          	push   -0x8(%r10)
    23ed:	55                   	push   %rbp
    23ee:	48 89 e5             	mov    %rsp,%rbp
    23f1:	41 57                	push   %r15
    23f3:	41 56                	push   %r14
    23f5:	41 55                	push   %r13
    23f7:	41 54                	push   %r12
    23f9:	41 52                	push   %r10
    23fb:	53                   	push   %rbx
    23fc:	48 81 ec e0 02 00 00 	sub    $0x2e0,%rsp
    if (argc < 2)
    2403:	83 ff 01             	cmp    $0x1,%edi
    2406:	0f 8e d7 06 00 00    	jle    2ae3 <main+0x703>
        std::cerr << "Usage: ./fft_cpu <input_path>" << std::endl;
        return 1;
    }

    // init data and output
    std::vector<std::complex<float>> data = read_complex_data(argv[1]);
    240c:	4c 8b 6e 08          	mov    0x8(%rsi),%r13
	: allocator_type(__a), _M_p(__dat) { }
    2410:	4c 8d b5 e0 fd ff ff 	lea    -0x220(%rbp),%r14
    2417:	4c 8d a5 d0 fd ff ff 	lea    -0x230(%rbp),%r12
    241e:	4c 89 b5 d0 fd ff ff 	mov    %r14,-0x230(%rbp)
	if (__s == 0)
    2425:	4d 85 ed             	test   %r13,%r13
    2428:	0f 84 97 fe ff ff    	je     22c5 <main.cold>
      {
#if __cplusplus >= 201703L
	if (std::__is_constant_evaluated())
	  return __gnu_cxx::char_traits<char_type>::length(__s);
#endif
	return __builtin_strlen(__s);
    242e:	4c 89 ef             	mov    %r13,%rdi
    2431:	e8 7a fc ff ff       	call   20b0 <strlen@plt>
      void
      basic_string<_CharT, _Traits, _Alloc>::
      _M_construct(_InIterator __beg, _InIterator __end,
		   std::forward_iterator_tag)
      {
	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
    2436:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
    243d:	48 89 c3             	mov    %rax,%rbx

	if (__dnew > size_type(_S_local_capacity))
    2440:	48 83 f8 0f          	cmp    $0xf,%rax
    2444:	0f 87 3d 06 00 00    	ja     2a87 <main+0x6a7>
	if (__n == 1)
    244a:	48 83 f8 01          	cmp    $0x1,%rax
    244e:	0f 85 bc 06 00 00    	jne    2b10 <main+0x730>
	__c1 = __c2;
    2454:	41 0f b6 45 00       	movzbl 0x0(%r13),%eax
    2459:	88 85 e0 fd ff ff    	mov    %al,-0x220(%rbp)

	this->_S_copy_chars(_M_data(), __beg, __end);

	__guard._M_guarded = 0;

	_M_set_length(__dnew);
    245f:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    2466:	48 8b 95 d0 fd ff ff 	mov    -0x230(%rbp),%rdx
    246d:	4c 8d b5 50 fd ff ff 	lea    -0x2b0(%rbp),%r14
    2474:	4c 89 e6             	mov    %r12,%rsi
    2477:	4c 89 f7             	mov    %r14,%rdi
      { _M_string_length = __length; }
    247a:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
    2481:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
    2485:	e8 f6 32 00 00       	call   5780 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>
    248a:	48 8b 85 60 fd ff ff 	mov    -0x2a0(%rbp),%rax
      { _M_dispose(); }
    2491:	4c 89 e7             	mov    %r12,%rdi
    2494:	48 89 85 08 fd ff ff 	mov    %rax,-0x2f8(%rbp)
    249b:	e8 f0 fc ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
      { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
    24a0:	48 8b 8d 50 fd ff ff 	mov    -0x2b0(%rbp),%rcx
    24a7:	48 8b 85 58 fd ff ff 	mov    -0x2a8(%rbp),%rax
    24ae:	48 29 c8             	sub    %rcx,%rax
    24b1:	48 89 8d 30 fd ff ff 	mov    %rcx,-0x2d0(%rbp)
    24b8:	48 89 c3             	mov    %rax,%rbx
    24bb:	48 89 85 28 fd ff ff 	mov    %rax,-0x2d8(%rbp)
    24c2:	48 89 c1             	mov    %rax,%rcx
    24c5:	48 89 85 38 fd ff ff 	mov    %rax,-0x2c8(%rbp)
    24cc:	48 c1 fb 03          	sar    $0x3,%rbx
	if (__n > _S_max_size(_Tp_alloc_type(__a)))
    24d0:	48 b8 f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%rax
    24d7:	ff ff 7f 
      { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
    24da:	49 89 df             	mov    %rbx,%r15
	if (__n > _S_max_size(_Tp_alloc_type(__a)))
    24dd:	48 39 c8             	cmp    %rcx,%rax
    24e0:	0f 82 6f fe ff ff    	jb     2355 <main.cold+0x90>
	: _M_start(), _M_finish(), _M_end_of_storage()
    24e6:	48 c7 85 78 fd ff ff 	movq   $0x0,-0x288(%rbp)
    24ed:	00 00 00 00 
	return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
    24f1:	48 85 db             	test   %rbx,%rbx
    24f4:	0f 84 c6 05 00 00    	je     2ac0 <main+0x6e0>
	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
    24fa:	4c 8b ad 28 fd ff ff 	mov    -0x2d8(%rbp),%r13
    2501:	4c 89 ef             	mov    %r13,%rdi
    2504:	e8 37 fc ff ff       	call   2140 <_Znwm@plt>
	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
    2509:	4a 8d 3c 28          	lea    (%rax,%r13,1),%rdi
    250d:	48 89 85 40 fd ff ff 	mov    %rax,-0x2c0(%rbp)
    2514:	48 89 c1             	mov    %rax,%rcx
    2517:	48 8d 43 ff          	lea    -0x1(%rbx),%rax
    251b:	48 89 bd 20 fd ff ff 	mov    %rdi,-0x2e0(%rbp)
        __uninit_default_n(_ForwardIterator __first, _Size __n)
        {
	  _ForwardIterator __cur = __first;
	  __try
	    {
	      for (; __n > 0; --__n, (void) ++__cur)
    2522:	48 83 f8 02          	cmp    $0x2,%rax
    2526:	0f 86 1c 06 00 00    	jbe    2b48 <main+0x768>
    252c:	48 89 de             	mov    %rbx,%rsi
    252f:	48 89 c8             	mov    %rcx,%rax

      _GLIBCXX_CONSTEXPR complex(_ComplexT __z) : _M_value(__z) { }

      _GLIBCXX_CONSTEXPR complex(float __r = 0.0f, float __i = 0.0f)
#if __cplusplus >= 201103L
      : _M_value{ __r, __i } { }
    2532:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    2536:	48 c1 ee 02          	shr    $0x2,%rsi
    253a:	48 c1 e6 05          	shl    $0x5,%rsi
    253e:	48 8d 14 0e          	lea    (%rsi,%rcx,1),%rdx
    2542:	40 80 e6 20          	and    $0x20,%sil
    2546:	74 18                	je     2560 <main+0x180>
    2548:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    254f:	c5 fc 11 00          	vmovups %ymm0,(%rax)
    2553:	48 83 c0 20          	add    $0x20,%rax
    2557:	48 39 d0             	cmp    %rdx,%rax
    255a:	74 16                	je     2572 <main+0x192>
    255c:	0f 1f 40 00          	nopl   0x0(%rax)
    2560:	c5 fc 11 00          	vmovups %ymm0,(%rax)
      _GLIBCXX_CONSTEXPR complex(float __r = 0.0f, float __i = 0.0f)
    2564:	48 83 c0 40          	add    $0x40,%rax
      : _M_value{ __r, __i } { }
    2568:	c5 fc 11 40 e0       	vmovups %ymm0,-0x20(%rax)
    256d:	48 39 d0             	cmp    %rdx,%rax
    2570:	75 ee                	jne    2560 <main+0x180>
    2572:	48 8b 8d 40 fd ff ff 	mov    -0x2c0(%rbp),%rcx
    2579:	48 89 d8             	mov    %rbx,%rax
    257c:	48 83 e0 fc          	and    $0xfffffffffffffffc,%rax
    2580:	48 8d 34 c1          	lea    (%rcx,%rax,8),%rsi
    2584:	48 39 c3             	cmp    %rax,%rbx
    2587:	0f 84 b3 05 00 00    	je     2b40 <main+0x760>
    258d:	c5 f8 77             	vzeroupper
    2590:	48 89 da             	mov    %rbx,%rdx
    2593:	48 29 c2             	sub    %rax,%rdx
    2596:	48 83 fa 01          	cmp    $0x1,%rdx
    259a:	74 1d                	je     25b9 <main+0x1d9>
    259c:	48 8b 8d 40 fd ff ff 	mov    -0x2c0(%rbp),%rcx
    25a3:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    25a7:	c5 f8 11 04 c1       	vmovups %xmm0,(%rcx,%rax,8)
    25ac:	f6 c2 01             	test   $0x1,%dl
    25af:	74 0d                	je     25be <main+0x1de>
    25b1:	48 83 e2 fe          	and    $0xfffffffffffffffe,%rdx
    25b5:	48 8d 34 d6          	lea    (%rsi,%rdx,8),%rsi
    25b9:	31 c9                	xor    %ecx,%ecx
    25bb:	48 89 0e             	mov    %rcx,(%rsi)
    25be:	48 89 bd 48 fd ff ff 	mov    %rdi,-0x2b8(%rbp)
	this->_M_impl._M_finish =
    25c5:	48 8b 85 48 fd ff ff 	mov    -0x2b8(%rbp),%rax
    25cc:	48 89 85 78 fd ff ff 	mov    %rax,-0x288(%rbp)
    size_t N = data.size();
    std::vector<std::complex<float>> out(N);

    // fft_cooley_tukey(N, data, out);
    auto start = std::chrono::high_resolution_clock::now();
    25d3:	e8 68 fa ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    fft_stockham_recursive(N, data, out);
    25d8:	48 89 df             	mov    %rbx,%rdi
    25db:	48 8d 95 70 fd ff ff 	lea    -0x290(%rbp),%rdx
    25e2:	4c 89 f6             	mov    %r14,%rsi
    auto start = std::chrono::high_resolution_clock::now();
    25e5:	49 89 c5             	mov    %rax,%r13
    fft_stockham_recursive(N, data, out);
    25e8:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    25ef:	48 89 85 70 fd ff ff 	mov    %rax,-0x290(%rbp)
    25f6:	48 8b 85 20 fd ff ff 	mov    -0x2e0(%rbp),%rax
    25fd:	48 89 85 80 fd ff ff 	mov    %rax,-0x280(%rbp)
    2604:	e8 37 14 00 00       	call   3a40 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_>

    auto end = std::chrono::high_resolution_clock::now();
    2609:	e8 32 fa ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
	  static constexpr _ToDur
	  __cast(const duration<_Rep, _Period>& __d)
	  {
	    typedef typename _ToDur::rep			__to_rep;
	    return _ToDur(static_cast<__to_rep>(
	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
    260e:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
      __ostream_type&
      operator<<(float __f)
      {
	// _GLIBCXX_RESOLVE_LIB_DEFECTS
	// 117. basic_ostream uses nonexistent num_put member functions.
	return _M_insert(static_cast<double>(__f));
    2612:	48 8d 3d 67 5b 00 00 	lea    0x5b67(%rip),%rdi        # 8180 <_ZSt4cout@GLIBCXX_3.4>
		const duration<_Rep2, _Period2>& __rhs)
      {
	typedef duration<_Rep1, _Period1>			__dur1;
	typedef duration<_Rep2, _Period2>			__dur2;
	typedef typename common_type<__dur1,__dur2>::type	__cd;
	return __cd(__cd(__lhs).count() - __cd(__rhs).count());
    2619:	4c 29 e8             	sub    %r13,%rax
	      static_cast<_CR>(__d.count()) / static_cast<_CR>(_CF::den)));
    261c:	c4 e1 ea 2a c0       	vcvtsi2ss %rax,%xmm2,%xmm0
    2621:	c5 fa 5e 05 e7 39 00 	vdivss 0x39e7(%rip),%xmm0,%xmm0        # 6010 <_IO_stdin_used+0x10>
    2628:	00 
    2629:	c5 fa 5a c0          	vcvtss2sd %xmm0,%xmm0,%xmm0
    262d:	e8 ce fb ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2632:	48 89 c7             	mov    %rax,%rdi
	return __pf(*this);
    2635:	e8 86 06 00 00       	call   2cc0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
      const unsigned __b2 = __base  * __base;
      const unsigned __b3 = __b2 * __base;
      const unsigned long __b4 = __b3 * __base;
      for (;;)
	{
	  if (__value < (unsigned)__base) return __n;
    263a:	48 83 bd 38 fd ff ff 	cmpq   $0x48,-0x2c8(%rbp)
    2641:	48 
    2642:	41 bd 01 00 00 00    	mov    $0x1,%r13d
	  if (__value < __b2) return __n + 1;
	  if (__value < __b3) return __n + 2;
	  if (__value < __b4) return __n + 3;
	  __value /= __b4;
    2648:	48 be 4b 59 86 38 d6 	movabs $0x346dc5d63886594b,%rsi
    264f:	c5 6d 34 
	  if (__value < (unsigned)__base) return __n;
    2652:	77 40                	ja     2694 <main+0x2b4>
    2654:	eb 47                	jmp    269d <main+0x2bd>
    2656:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    265d:	00 00 00 
	  if (__value < __b3) return __n + 2;
    2660:	48 81 fb e7 03 00 00 	cmp    $0x3e7,%rbx
    2667:	0f 86 c1 04 00 00    	jbe    2b2e <main+0x74e>
	  if (__value < __b4) return __n + 3;
    266d:	48 81 fb 0f 27 00 00 	cmp    $0x270f,%rbx
    2674:	0f 86 bd 04 00 00    	jbe    2b37 <main+0x757>
	  __value /= __b4;
    267a:	48 89 d8             	mov    %rbx,%rax
	  __n += 4;
    267d:	41 83 c5 04          	add    $0x4,%r13d
	  __value /= __b4;
    2681:	48 f7 e6             	mul    %rsi
    2684:	48 c1 ea 0b          	shr    $0xb,%rdx
	  if (__value < (unsigned)__base) return __n;
    2688:	48 81 fb 9f 86 01 00 	cmp    $0x1869f,%rbx
    268f:	76 0c                	jbe    269d <main+0x2bd>
    2691:	48 89 d3             	mov    %rdx,%rbx
	  if (__value < __b2) return __n + 1;
    2694:	48 83 fb 63          	cmp    $0x63,%rbx
    2698:	77 c6                	ja     2660 <main+0x280>
    269a:	41 ff c5             	inc    %r13d
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    269d:	48 8d 9d b0 fd ff ff 	lea    -0x250(%rbp),%rbx
  noexcept // any 32-bit value fits in the SSO buffer
#endif
  {
    const auto __len = __detail::__to_chars_len(__val);
    string __str;
    __str.__resize_and_overwrite(__len, [__val](char* __p, size_t __n) {
    26a4:	45 89 ee             	mov    %r13d,%r14d
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    26a7:	48 8d 85 c0 fd ff ff 	lea    -0x240(%rbp),%rax
      { _M_string_length = __length; }
    26ae:	48 c7 85 b8 fd ff ff 	movq   $0x0,-0x248(%rbp)
    26b5:	00 00 00 00 
    resize_and_overwrite(const size_type __n, _Operation __op)
#else
    __resize_and_overwrite(const size_type __n, _Operation __op)
#endif
    {
      reserve(__n);
    26b9:	4c 89 f6             	mov    %r14,%rsi
    26bc:	48 89 df             	mov    %rbx,%rdi
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    26bf:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
    26c6:	c6 85 c0 fd ff ff 00 	movb   $0x0,-0x240(%rbp)
    26cd:	e8 1e fb ff ff       	call   21f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@plt>
    {
#if __cpp_variable_templates
      static_assert(__integer_to_chars_is_unsigned<_Tp>, "implementation bug");
#endif

      constexpr char __digits[201] =
    26d2:	c5 fd 6f 05 46 3a 00 	vmovdqa 0x3a46(%rip),%ymm0        # 6120 <_IO_stdin_used+0x120>
    26d9:	00 
	"0001020304050607080910111213141516171819"
	"2021222324252627282930313233343536373839"
	"4041424344454647484950515253545556575859"
	"6061626364656667686970717273747576777879"
	"8081828384858687888990919293949596979899";
      unsigned __pos = __len - 1;
    26da:	41 ff cd             	dec    %r13d
      while (__val >= 100)
    26dd:	48 81 bd 38 fd ff ff 	cmpq   $0x318,-0x2c8(%rbp)
    26e4:	18 03 00 00 
      { return _M_dataplus._M_p; }
    26e8:	48 8b bd b0 fd ff ff 	mov    -0x250(%rbp),%rdi
      constexpr char __digits[201] =
    26ef:	c5 fd 7f 85 d0 fd ff 	vmovdqa %ymm0,-0x230(%rbp)
    26f6:	ff 
    26f7:	c5 fd 6f 05 41 3a 00 	vmovdqa 0x3a41(%rip),%ymm0        # 6140 <_IO_stdin_used+0x140>
    26fe:	00 
    26ff:	c5 fd 7f 85 f0 fd ff 	vmovdqa %ymm0,-0x210(%rbp)
    2706:	ff 
    2707:	c5 fd 6f 05 51 3a 00 	vmovdqa 0x3a51(%rip),%ymm0        # 6160 <_IO_stdin_used+0x160>
    270e:	00 
    270f:	c5 fd 7f 85 10 fe ff 	vmovdqa %ymm0,-0x1f0(%rbp)
    2716:	ff 
    2717:	c5 fd 6f 05 61 3a 00 	vmovdqa 0x3a61(%rip),%ymm0        # 6180 <_IO_stdin_used+0x180>
    271e:	00 
    271f:	c5 fd 7f 85 30 fe ff 	vmovdqa %ymm0,-0x1d0(%rbp)
    2726:	ff 
    2727:	c5 fd 6f 05 71 3a 00 	vmovdqa 0x3a71(%rip),%ymm0        # 61a0 <_IO_stdin_used+0x1a0>
    272e:	00 
    272f:	c5 fd 7f 85 50 fe ff 	vmovdqa %ymm0,-0x1b0(%rbp)
    2736:	ff 
    2737:	c5 fd 6f 05 81 3a 00 	vmovdqa 0x3a81(%rip),%ymm0        # 61c0 <_IO_stdin_used+0x1c0>
    273e:	00 
    273f:	c5 fd 7f 85 70 fe ff 	vmovdqa %ymm0,-0x190(%rbp)
    2746:	ff 
    2747:	c5 f9 6f 05 b1 39 00 	vmovdqa 0x39b1(%rip),%xmm0        # 6100 <_IO_stdin_used+0x100>
    274e:	00 
    274f:	c5 fa 7f 85 89 fe ff 	vmovdqu %xmm0,-0x177(%rbp)
    2756:	ff 
      while (__val >= 100)
    2757:	76 5f                	jbe    27b8 <main+0x3d8>
	{
	  auto const __num = (__val % 100) * 2;
    2759:	48 be c3 f5 28 5c 8f 	movabs $0x28f5c28f5c28f5c3,%rsi
    2760:	c2 f5 28 
    2763:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2768:	4c 89 fa             	mov    %r15,%rdx
    276b:	48 c1 ea 02          	shr    $0x2,%rdx
    276f:	48 89 d0             	mov    %rdx,%rax
    2772:	48 f7 e6             	mul    %rsi
    2775:	4c 89 f8             	mov    %r15,%rax
    2778:	48 c1 ea 02          	shr    $0x2,%rdx
    277c:	48 6b ca 64          	imul   $0x64,%rdx,%rcx
    2780:	48 29 c8             	sub    %rcx,%rax
    2783:	4c 89 f9             	mov    %r15,%rcx
	  __val /= 100;
    2786:	49 89 d7             	mov    %rdx,%r15
	  __first[__pos] = __digits[__num + 1];
    2789:	44 89 ea             	mov    %r13d,%edx
	  auto const __num = (__val % 100) * 2;
    278c:	48 01 c0             	add    %rax,%rax
	  __first[__pos] = __digits[__num + 1];
    278f:	44 0f b6 84 05 d1 fd 	movzbl -0x22f(%rbp,%rax,1),%r8d
    2796:	ff ff 
    2798:	44 88 04 17          	mov    %r8b,(%rdi,%rdx,1)
	  __first[__pos - 1] = __digits[__num];
    279c:	0f b6 84 05 d0 fd ff 	movzbl -0x230(%rbp,%rax,1),%eax
    27a3:	ff 
    27a4:	41 8d 55 ff          	lea    -0x1(%r13),%edx
	  __pos -= 2;
    27a8:	41 83 ed 02          	sub    $0x2,%r13d
	  __first[__pos - 1] = __digits[__num];
    27ac:	88 04 17             	mov    %al,(%rdi,%rdx,1)
      while (__val >= 100)
    27af:	48 81 f9 0f 27 00 00 	cmp    $0x270f,%rcx
    27b6:	77 b0                	ja     2768 <main+0x388>
	  auto const __num = __val * 2;
	  __first[1] = __digits[__num + 1];
	  __first[0] = __digits[__num];
	}
      else
	__first[0] = '0' + __val;
    27b8:	41 8d 47 30          	lea    0x30(%r15),%eax
      if (__val >= 10)
    27bc:	49 83 ff 09          	cmp    $0x9,%r15
    27c0:	76 17                	jbe    27d9 <main+0x3f9>
	  auto const __num = __val * 2;
    27c2:	4b 8d 0c 3f          	lea    (%r15,%r15,1),%rcx
	  __first[1] = __digits[__num + 1];
    27c6:	0f b6 84 0d d1 fd ff 	movzbl -0x22f(%rbp,%rcx,1),%eax
    27cd:	ff 
    27ce:	88 47 01             	mov    %al,0x1(%rdi)
	  __first[0] = __digits[__num];
    27d1:	0f b6 84 0d d0 fd ff 	movzbl -0x230(%rbp,%rcx,1),%eax
    27d8:	ff 
    27d9:	88 07                	mov    %al,(%rdi)
    27db:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    { return std::move(__rhs.insert(0, __lhs)); }
    27e2:	31 f6                	xor    %esi,%esi
    27e4:	48 89 df             	mov    %rbx,%rdi
      { _M_string_length = __length; }
    27e7:	4c 89 b5 b8 fd ff ff 	mov    %r14,-0x248(%rbp)
    { return std::move(__rhs.insert(0, __lhs)); }
    27ee:	48 8d 15 57 38 00 00 	lea    0x3857(%rip),%rdx        # 604c <_IO_stdin_used+0x4c>
    27f5:	42 c6 04 30 00       	movb   $0x0,(%rax,%r14,1)
    27fa:	c5 f8 77             	vzeroupper
    27fd:	e8 5e f8 ff ff       	call   2060 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@plt>
    2802:	48 89 c6             	mov    %rax,%rsi
    2805:	4c 89 e7             	mov    %r12,%rdi
    2808:	e8 c3 f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
    { return std::move(__lhs.append(__rhs)); }
    280d:	48 8d 35 4d 38 00 00 	lea    0x384d(%rip),%rsi        # 6061 <_IO_stdin_used+0x61>
    2814:	4c 89 e7             	mov    %r12,%rdi
    2817:	e8 74 fa ff ff       	call   2290 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>
    281c:	48 89 c6             	mov    %rax,%rsi
    281f:	48 8d 85 90 fd ff ff 	lea    -0x270(%rbp),%rax
       *  The default constructor does nothing and is not normally
       *  accessible to users.
      */
      basic_ios()
      : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
    2826:	4c 8d b5 c8 fe ff ff 	lea    -0x138(%rbp),%r14
    282d:	48 89 c7             	mov    %rax,%rdi
    2830:	48 89 85 20 fd ff ff 	mov    %rax,-0x2e0(%rbp)
    2837:	e8 94 f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
      { _M_dispose(); }
    283c:	4c 89 e7             	mov    %r12,%rdi
    283f:	e8 4c f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2844:	48 89 df             	mov    %rbx,%rdi
    2847:	e8 44 f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    284c:	4c 89 f7             	mov    %r14,%rdi
    284f:	4c 89 b5 10 fd ff ff 	mov    %r14,-0x2f0(%rbp)
    2856:	e8 25 f8 ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
    285b:	48 8d 05 56 53 00 00 	lea    0x5356(%rip),%rax        # 7bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2862:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
       __ostream_type&
      seekp(off_type, ios_base::seekdir);

    protected:
      basic_ostream()
      { this->init(0); }
    2866:	31 f6                	xor    %esi,%esi
    2868:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
      : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
    286f:	31 c0                	xor    %eax,%eax
    2871:	66 89 45 a8          	mov    %ax,-0x58(%rbp)
    2875:	48 8b 05 d4 53 00 00 	mov    0x53d4(%rip),%rax        # 7c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
    287c:	c5 fd 7f 45 b0       	vmovdqa %ymm0,-0x50(%rbp)
    2881:	48 8b 78 e8          	mov    -0x18(%rax),%rdi
    2885:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    288c:	48 8b 05 c5 53 00 00 	mov    0x53c5(%rip),%rax        # 7c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
      : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
    2893:	48 c7 45 a0 00 00 00 	movq   $0x0,-0x60(%rbp)
    289a:	00 
    289b:	4c 01 e7             	add    %r12,%rdi
    289e:	48 89 07             	mov    %rax,(%rdi)
    28a1:	c5 f8 77             	vzeroupper
    28a4:	e8 07 f9 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
      : __ostream_type(), _M_filebuf()
    28a9:	48 8d 05 a0 54 00 00 	lea    0x54a0(%rip),%rax        # 7d50 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    28b0:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    28b7:	48 83 c0 28          	add    $0x28,%rax
    28bb:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    28c2:	48 8d 85 d8 fd ff ff 	lea    -0x228(%rbp),%rax
    28c9:	48 89 c7             	mov    %rax,%rdi
    28cc:	48 89 85 18 fd ff ff 	mov    %rax,-0x2e8(%rbp)
    28d3:	48 89 c3             	mov    %rax,%rbx
    28d6:	e8 85 f8 ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
	this->init(&_M_filebuf);
    28db:	48 89 de             	mov    %rbx,%rsi
    28de:	4c 89 f7             	mov    %r14,%rdi
    28e1:	e8 ca f8 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
      { return open(__s.c_str(), __mode); }
    28e6:	48 8b b5 90 fd ff ff 	mov    -0x270(%rbp),%rsi
    28ed:	ba 10 00 00 00       	mov    $0x10,%edx
    28f2:	48 89 df             	mov    %rbx,%rdi
    28f5:	e8 36 f8 ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
       */
      void
      open(const std::string& __s, ios_base::openmode __mode = ios_base::out)
      {
	if (!_M_filebuf.open(__s, __mode | ios_base::out))
	  this->setstate(ios_base::failbit);
    28fa:	48 8b 95 d0 fd ff ff 	mov    -0x230(%rbp),%rdx
    2901:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    2905:	4c 01 e7             	add    %r12,%rdi
	if (!_M_filebuf.open(__s, __mode | ios_base::out))
    2908:	48 85 c0             	test   %rax,%rax
    290b:	0f 84 0d 02 00 00    	je     2b1e <main+0x73e>
	else
	  // _GLIBCXX_RESOLVE_LIB_DEFECTS
	  // 409. Closing an fstream should clear error state
	  this->clear();
    2911:	31 f6                	xor    %esi,%esi
    2913:	e8 18 f9 ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    for (const auto &c : data)
    2918:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    operator<<(basic_ostream<char, _Traits>& __out, const char* __s)
    {
      if (!__s)
	__out.setstate(ios_base::badbit);
      else
	__ostream_insert(__out, __s,
    291f:	4c 8d 35 40 37 00 00 	lea    0x3740(%rip),%r14        # 6066 <_IO_stdin_used+0x66>
    2926:	4c 8d 2d 3b 37 00 00 	lea    0x373b(%rip),%r13        # 6068 <_IO_stdin_used+0x68>
    292d:	48 89 c3             	mov    %rax,%rbx
    2930:	48 39 85 48 fd ff ff 	cmp    %rax,-0x2b8(%rbp)
    2937:	74 58                	je     2991 <main+0x5b1>
    2939:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
	return _M_insert(static_cast<double>(__f));
    2940:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    2944:	4c 89 e7             	mov    %r12,%rdi
    2947:	c5 f2 5a 03          	vcvtss2sd (%rbx),%xmm1,%xmm0
    294b:	e8 b0 f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
	__ostream_insert(__out, __s,
    2950:	ba 01 00 00 00       	mov    $0x1,%edx
    2955:	4c 89 f6             	mov    %r14,%rsi
    2958:	48 89 c7             	mov    %rax,%rdi
	return _M_insert(static_cast<double>(__f));
    295b:	49 89 c7             	mov    %rax,%r15
	__ostream_insert(__out, __s,
    295e:	e8 0d f8 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
	return _M_insert(static_cast<double>(__f));
    2963:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    2967:	4c 89 ff             	mov    %r15,%rdi
    296a:	c5 f2 5a 43 04       	vcvtss2sd 0x4(%rbx),%xmm1,%xmm0
    296f:	e8 8c f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2974:	48 89 c7             	mov    %rax,%rdi
	__ostream_insert(__out, __s,
    2977:	ba 01 00 00 00       	mov    $0x1,%edx
    297c:	4c 89 ee             	mov    %r13,%rsi
    297f:	e8 ec f7 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2984:	48 83 c3 08          	add    $0x8,%rbx
    2988:	48 39 9d 48 fd ff ff 	cmp    %rbx,-0x2b8(%rbp)
    298f:	75 af                	jne    2940 <main+0x560>
      { }
    2991:	48 8d 05 e0 53 00 00 	lea    0x53e0(%rip),%rax        # 7d78 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x40>
    2998:	c5 fa 7e 1d 08 54 00 	vmovq  0x5408(%rip),%xmm3        # 7da8 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x70>
    299f:	00 
	  { this->close(); }
    29a0:	48 8b bd 18 fd ff ff 	mov    -0x2e8(%rbp),%rdi
      { }
    29a7:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    29ae:	48 8d 05 13 53 00 00 	lea    0x5313(%rip),%rax        # 7cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29b5:	c4 e3 e1 22 c0 01    	vpinsrq $0x1,%rax,%xmm3,%xmm0
    29bb:	c5 f9 7f 85 d0 fd ff 	vmovdqa %xmm0,-0x230(%rbp)
    29c2:	ff 
	  { this->close(); }
    29c3:	e8 88 f6 ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
      }
    29c8:	48 8d bd 40 fe ff ff 	lea    -0x1c0(%rbp),%rdi
    29cf:	e8 9c f8 ff ff       	call   2270 <_ZNSt12__basic_fileIcED1Ev@plt>

  public:
      /// Destructor deallocates no buffer space.
      virtual
      ~basic_streambuf()
      { }
    29d4:	48 8d 05 fd 51 00 00 	lea    0x51fd(%rip),%rax        # 7bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29db:	48 8d bd 10 fe ff ff 	lea    -0x1f0(%rbp),%rdi
    29e2:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
    29e9:	e8 d2 f7 ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
      ~basic_ostream() { }
    29ee:	48 8b 05 5b 52 00 00 	mov    0x525b(%rip),%rax        # 7c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    29f5:	48 8b 0d 5c 52 00 00 	mov    0x525c(%rip),%rcx        # 7c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
      ~basic_ios() { }
    29fc:	48 8b bd 10 fd ff ff 	mov    -0x2f0(%rbp),%rdi
    2a03:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    2a0a:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2a0e:	48 89 8c 05 d0 fd ff 	mov    %rcx,-0x230(%rbp,%rax,1)
    2a15:	ff 
    2a16:	48 8d 05 9b 51 00 00 	lea    0x519b(%rip),%rax        # 7bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2a1d:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    2a24:	e8 67 f6 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    2a29:	48 8b bd 20 fd ff ff 	mov    -0x2e0(%rbp),%rdi
    2a30:	e8 5b f7 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
	if (__p)
    2a35:	48 83 bd 40 fd ff ff 	cmpq   $0x0,-0x2c0(%rbp)
    2a3c:	00 
    2a3d:	74 13                	je     2a52 <main+0x672>
	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
    2a3f:	48 8b b5 38 fd ff ff 	mov    -0x2c8(%rbp),%rsi
    2a46:	48 8b bd 40 fd ff ff 	mov    -0x2c0(%rbp),%rdi
    2a4d:	e8 fe f6 ff ff       	call   2150 <_ZdlPvm@plt>
    2a52:	48 8b bd 30 fd ff ff 	mov    -0x2d0(%rbp),%rdi
    2a59:	48 85 ff             	test   %rdi,%rdi
    2a5c:	74 0f                	je     2a6d <main+0x68d>
		      _M_impl._M_end_of_storage - _M_impl._M_start);
    2a5e:	48 8b b5 08 fd ff ff 	mov    -0x2f8(%rbp),%rsi
    2a65:	48 29 fe             	sub    %rdi,%rsi
    2a68:	e8 e3 f6 ff ff       	call   2150 <_ZdlPvm@plt>
    std::cout << diff.count() << std::endl;

    std::string out_file = "data/output_fft_cpu_" + std::to_string(N) + ".txt";
    write_complex_data(out, out_file);

    return 0;
    2a6d:	31 c0                	xor    %eax,%eax
    2a6f:	48 81 c4 e0 02 00 00 	add    $0x2e0,%rsp
    2a76:	5b                   	pop    %rbx
    2a77:	41 5a                	pop    %r10
    2a79:	41 5c                	pop    %r12
    2a7b:	41 5d                	pop    %r13
    2a7d:	41 5e                	pop    %r14
    2a7f:	41 5f                	pop    %r15
    2a81:	5d                   	pop    %rbp
    2a82:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    2a86:	c3                   	ret
	    _M_data(_M_create(__dnew, size_type(0)));
    2a87:	4c 89 e7             	mov    %r12,%rdi
    2a8a:	48 8d b5 b0 fd ff ff 	lea    -0x250(%rbp),%rsi
    2a91:	31 d2                	xor    %edx,%edx
    2a93:	e8 c8 f7 ff ff       	call   2260 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
      { _M_dataplus._M_p = __p; }
    2a98:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    2a9f:	48 89 c7             	mov    %rax,%rdi
      { _M_allocated_capacity = __capacity; }
    2aa2:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    2aa9:	48 89 85 e0 fd ff ff 	mov    %rax,-0x220(%rbp)
	  return __s1;
#if __cplusplus >= 202002L
	if (std::__is_constant_evaluated())
	  return __gnu_cxx::char_traits<char_type>::copy(__s1, __s2, __n);
#endif
	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
    2ab0:	48 89 da             	mov    %rbx,%rdx
    2ab3:	4c 89 ee             	mov    %r13,%rsi
    2ab6:	e8 55 f6 ff ff       	call   2110 <memcpy@plt>
    2abb:	e9 9f f9 ff ff       	jmp    245f <main+0x7f>
	  _ForwardIterator __cur = __first;
    2ac0:	31 d2                	xor    %edx,%edx
    2ac2:	48 89 95 28 fd ff ff 	mov    %rdx,-0x2d8(%rbp)
    2ac9:	48 89 95 20 fd ff ff 	mov    %rdx,-0x2e0(%rbp)
    2ad0:	48 89 95 40 fd ff ff 	mov    %rdx,-0x2c0(%rbp)
    2ad7:	48 89 95 48 fd ff ff 	mov    %rdx,-0x2b8(%rbp)
    2ade:	e9 e2 fa ff ff       	jmp    25c5 <main+0x1e5>
	__ostream_insert(__out, __s,
    2ae3:	48 8d 1d b6 57 00 00 	lea    0x57b6(%rip),%rbx        # 82a0 <_ZSt4cerr@GLIBCXX_3.4>
    2aea:	ba 1d 00 00 00       	mov    $0x1d,%edx
    2aef:	48 8d 35 38 35 00 00 	lea    0x3538(%rip),%rsi        # 602e <_IO_stdin_used+0x2e>
    2af6:	48 89 df             	mov    %rbx,%rdi
    2af9:	e8 72 f6 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
	return __pf(*this);
    2afe:	48 89 df             	mov    %rbx,%rdi
    2b01:	e8 ba 01 00 00       	call   2cc0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
        return 1;
    2b06:	b8 01 00 00 00       	mov    $0x1,%eax
    2b0b:	e9 5f ff ff ff       	jmp    2a6f <main+0x68f>
	if (__n == 0)
    2b10:	48 85 c0             	test   %rax,%rax
    2b13:	0f 84 46 f9 ff ff    	je     245f <main+0x7f>
    2b19:	4c 89 f7             	mov    %r14,%rdi
    2b1c:	eb 92                	jmp    2ab0 <main+0x6d0>
  { return _Ios_Iostate(static_cast<int>(__a) & static_cast<int>(__b)); }

  _GLIBCXX_NODISCARD _GLIBCXX_CONSTEXPR
  inline _Ios_Iostate
  operator|(_Ios_Iostate __a, _Ios_Iostate __b) _GLIBCXX_NOTHROW
  { return _Ios_Iostate(static_cast<int>(__a) | static_cast<int>(__b)); }
    2b1e:	8b 77 20             	mov    0x20(%rdi),%esi
    2b21:	83 ce 04             	or     $0x4,%esi
      { this->clear(this->rdstate() | __state); }
    2b24:	e8 07 f7 ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    2b29:	e9 ea fd ff ff       	jmp    2918 <main+0x538>
	  if (__value < __b3) return __n + 2;
    2b2e:	41 83 c5 02          	add    $0x2,%r13d
    2b32:	e9 66 fb ff ff       	jmp    269d <main+0x2bd>
	  if (__value < __b4) return __n + 3;
    2b37:	41 83 c5 03          	add    $0x3,%r13d
    2b3b:	e9 5d fb ff ff       	jmp    269d <main+0x2bd>
    2b40:	c5 f8 77             	vzeroupper
    2b43:	e9 76 fa ff ff       	jmp    25be <main+0x1de>
    2b48:	48 8b b5 40 fd ff ff 	mov    -0x2c0(%rbp),%rsi
	this->_M_impl._M_end_of_storage = this->_M_impl._M_start + __n;
    2b4f:	31 c0                	xor    %eax,%eax
    2b51:	e9 3a fa ff ff       	jmp    2590 <main+0x1b0>
    2b56:	e9 76 f7 ff ff       	jmp    22d1 <main.cold+0xc>
	if (__p)
    2b5b:	49 89 c4             	mov    %rax,%r12
    2b5e:	c5 f8 77             	vzeroupper
    2b61:	e9 a2 f7 ff ff       	jmp    2308 <main.cold+0x43>
    2b66:	48 89 c3             	mov    %rax,%rbx
    2b69:	e9 b9 f7 ff ff       	jmp    2327 <main.cold+0x62>
	__catch(...)
    2b6e:	48 89 c7             	mov    %rax,%rdi
    2b71:	e9 cd f7 ff ff       	jmp    2343 <main.cold+0x7e>
      { _M_dispose(); }
    2b76:	48 89 c3             	mov    %rax,%rbx
    2b79:	e9 e3 f7 ff ff       	jmp    2361 <main.cold+0x9c>
      ~basic_ostream() { }
    2b7e:	49 89 c4             	mov    %rax,%r12
    2b81:	c5 f8 77             	vzeroupper
    2b84:	e9 fa f7 ff ff       	jmp    2383 <main.cold+0xbe>
      }
    2b89:	49 89 c4             	mov    %rax,%r12
    2b8c:	e9 e3 f7 ff ff       	jmp    2374 <main.cold+0xaf>
    2b91:	49 89 c4             	mov    %rax,%r12
    2b94:	e9 64 f7 ff ff       	jmp    22fd <main.cold+0x38>
    2b99:	49 89 c4             	mov    %rax,%r12
    2b9c:	c5 f8 77             	vzeroupper
    2b9f:	e9 2d f8 ff ff       	jmp    23d1 <main.cold+0x10c>
    2ba4:	49 89 c5             	mov    %rax,%r13
    2ba7:	e9 17 f8 ff ff       	jmp    23c3 <main.cold+0xfe>
      ~basic_ios() { }
    2bac:	49 89 c4             	mov    %rax,%r12
    2baf:	c5 f8 77             	vzeroupper
    2bb2:	e9 ed f7 ff ff       	jmp    23a4 <main.cold+0xdf>
    2bb7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2bbe:	00 00 

0000000000002bc0 <_start>:
    2bc0:	31 ed                	xor    %ebp,%ebp
    2bc2:	49 89 d1             	mov    %rdx,%r9
    2bc5:	5e                   	pop    %rsi
    2bc6:	48 89 e2             	mov    %rsp,%rdx
    2bc9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    2bcd:	50                   	push   %rax
    2bce:	54                   	push   %rsp
    2bcf:	45 31 c0             	xor    %r8d,%r8d
    2bd2:	31 c9                	xor    %ecx,%ecx
    2bd4:	48 8d 3d 05 f8 ff ff 	lea    -0x7fb(%rip),%rdi        # 23e0 <main>
    2bdb:	ff 15 e7 53 00 00    	call   *0x53e7(%rip)        # 7fc8 <__libc_start_main@GLIBC_2.34>
    2be1:	f4                   	hlt
    2be2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2be9:	00 00 00 
    2bec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002bf0 <deregister_tm_clones>:
    2bf0:	48 8d 3d 61 55 00 00 	lea    0x5561(%rip),%rdi        # 8158 <__TMC_END__>
    2bf7:	48 8d 05 5a 55 00 00 	lea    0x555a(%rip),%rax        # 8158 <__TMC_END__>
    2bfe:	48 39 f8             	cmp    %rdi,%rax
    2c01:	74 15                	je     2c18 <deregister_tm_clones+0x28>
    2c03:	48 8b 05 c6 53 00 00 	mov    0x53c6(%rip),%rax        # 7fd0 <_ITM_deregisterTMCloneTable@Base>
    2c0a:	48 85 c0             	test   %rax,%rax
    2c0d:	74 09                	je     2c18 <deregister_tm_clones+0x28>
    2c0f:	ff e0                	jmp    *%rax
    2c11:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2c18:	c3                   	ret
    2c19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002c20 <register_tm_clones>:
    2c20:	48 8d 3d 31 55 00 00 	lea    0x5531(%rip),%rdi        # 8158 <__TMC_END__>
    2c27:	48 8d 35 2a 55 00 00 	lea    0x552a(%rip),%rsi        # 8158 <__TMC_END__>
    2c2e:	48 29 fe             	sub    %rdi,%rsi
    2c31:	48 89 f0             	mov    %rsi,%rax
    2c34:	48 c1 ee 3f          	shr    $0x3f,%rsi
    2c38:	48 c1 f8 03          	sar    $0x3,%rax
    2c3c:	48 01 c6             	add    %rax,%rsi
    2c3f:	48 d1 fe             	sar    $1,%rsi
    2c42:	74 14                	je     2c58 <register_tm_clones+0x38>
    2c44:	48 8b 05 95 53 00 00 	mov    0x5395(%rip),%rax        # 7fe0 <_ITM_registerTMCloneTable@Base>
    2c4b:	48 85 c0             	test   %rax,%rax
    2c4e:	74 08                	je     2c58 <register_tm_clones+0x38>
    2c50:	ff e0                	jmp    *%rax
    2c52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2c58:	c3                   	ret
    2c59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002c60 <__do_global_dtors_aux>:
    2c60:	f3 0f 1e fa          	endbr64
    2c64:	80 3d 45 57 00 00 00 	cmpb   $0x0,0x5745(%rip)        # 83b0 <completed.0>
    2c6b:	75 2b                	jne    2c98 <__do_global_dtors_aux+0x38>
    2c6d:	55                   	push   %rbp
    2c6e:	48 83 3d 4a 53 00 00 	cmpq   $0x0,0x534a(%rip)        # 7fc0 <__cxa_finalize@GLIBC_2.2.5>
    2c75:	00 
    2c76:	48 89 e5             	mov    %rsp,%rbp
    2c79:	74 0c                	je     2c87 <__do_global_dtors_aux+0x27>
    2c7b:	48 8b 3d c6 54 00 00 	mov    0x54c6(%rip),%rdi        # 8148 <__dso_handle>
    2c82:	e8 29 f6 ff ff       	call   22b0 <__cxa_finalize@plt>
    2c87:	e8 64 ff ff ff       	call   2bf0 <deregister_tm_clones>
    2c8c:	c6 05 1d 57 00 00 01 	movb   $0x1,0x571d(%rip)        # 83b0 <completed.0>
    2c93:	5d                   	pop    %rbp
    2c94:	c3                   	ret
    2c95:	0f 1f 00             	nopl   (%rax)
    2c98:	c3                   	ret
    2c99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002ca0 <frame_dummy>:
    2ca0:	f3 0f 1e fa          	endbr64
    2ca4:	e9 77 ff ff ff       	jmp    2c20 <register_tm_clones>
    2ca9:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2cb0:	00 00 00 
    2cb3:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2cba:	00 00 00 
    2cbd:	0f 1f 00             	nopl   (%rax)

0000000000002cc0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>:
   *  https://gcc.gnu.org/onlinedocs/libstdc++/manual/streambufs.html#io.streambuf.buffering
   *  for more on this subject.
  */
  template<typename _CharT, typename _Traits>
    inline basic_ostream<_CharT, _Traits>&
    endl(basic_ostream<_CharT, _Traits>& __os)
    2cc0:	55                   	push   %rbp
    2cc1:	53                   	push   %rbx
    2cc2:	48 83 ec 08          	sub    $0x8,%rsp
    { return flush(__os.put(__os.widen('\n'))); }
    2cc6:	48 8b 07             	mov    (%rdi),%rax
    2cc9:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2ccd:	48 8b ac 07 f0 00 00 	mov    0xf0(%rdi,%rax,1),%rbp
    2cd4:	00 
      if (!__f)
    2cd5:	48 85 ed             	test   %rbp,%rbp
    2cd8:	0f 84 e2 f5 ff ff    	je     22c0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0.cold>
       *  @return  The converted character.
      */
      char_type
      widen(char __c) const
      {
	if (_M_widen_ok)
    2cde:	80 7d 38 00          	cmpb   $0x0,0x38(%rbp)
    2ce2:	48 89 fb             	mov    %rdi,%rbx
    2ce5:	74 1a                	je     2d01 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x41>
	  return _M_widen[static_cast<unsigned char>(__c)];
    2ce7:	0f be 75 43          	movsbl 0x43(%rbp),%esi
    2ceb:	48 89 df             	mov    %rbx,%rdi
    2cee:	e8 3d f3 ff ff       	call   2030 <_ZNSo3putEc@plt>
    2cf3:	48 83 c4 08          	add    $0x8,%rsp
    2cf7:	5b                   	pop    %rbx
    2cf8:	48 89 c7             	mov    %rax,%rdi
    2cfb:	5d                   	pop    %rbp
   *  This manipulator simply calls the stream's @c flush() member function.
  */
  template<typename _CharT, typename _Traits>
    inline basic_ostream<_CharT, _Traits>&
    flush(basic_ostream<_CharT, _Traits>& __os)
    { return __os.flush(); }
    2cfc:	e9 df f3 ff ff       	jmp    20e0 <_ZNSo5flushEv@plt>
	this->_M_widen_init();
    2d01:	48 89 ef             	mov    %rbp,%rdi
    2d04:	e8 77 f4 ff ff       	call   2180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
	return this->do_widen(__c);
    2d09:	48 8b 45 00          	mov    0x0(%rbp),%rax
    2d0d:	be 0a 00 00 00       	mov    $0xa,%esi
    2d12:	48 89 ef             	mov    %rbp,%rdi
    2d15:	ff 50 30             	call   *0x30(%rax)
    2d18:	0f be f0             	movsbl %al,%esi
    2d1b:	eb ce                	jmp    2ceb <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x2b>
    2d1d:	0f 1f 00             	nopl   (%rax)

0000000000002d20 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_>:
{
    2d20:	41 56                	push   %r14
    2d22:	41 55                	push   %r13
    2d24:	41 54                	push   %r12
    2d26:	55                   	push   %rbp
    2d27:	53                   	push   %rbx
    2d28:	48 83 ec 20          	sub    $0x20,%rsp
    for (int k = 0; k < N; k++)
    2d2c:	48 85 ff             	test   %rdi,%rdi
    2d2f:	0f 84 03 01 00 00    	je     2e38 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x118>
	return *(this->_M_impl._M_start + __n);
    2d35:	48 8b 1e             	mov    (%rsi),%rbx
    2d38:	4c 8b 2a             	mov    (%rdx),%r13
    2d3b:	48 89 fd             	mov    %rdi,%rbp
            float angle = 2.0f * M_PI * k * n / N;
    2d3e:	0f 88 01 01 00 00    	js     2e45 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x125>
    2d44:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
    2d48:	c4 e1 c3 2a c7       	vcvtsi2sd %rdi,%xmm7,%xmm0
    2d4d:	c5 f9 13 44 24 18    	vmovlpd %xmm0,0x18(%rsp)
    for (int k = 0; k < N; k++)
    2d53:	45 31 e4             	xor    %r12d,%r12d
    2d56:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2d5d:	00 00 00 
            float angle = 2.0f * M_PI * k * n / N;
    2d60:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    2d64:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
        for (int n = 0; n < N; n++)
    2d68:	45 31 f6             	xor    %r14d,%r14d
            float angle = 2.0f * M_PI * k * n / N;
    2d6b:	c4 c1 4b 2a c4       	vcvtsi2sd %r12d,%xmm6,%xmm0
    2d70:	c5 fb 59 15 68 33 00 	vmulsd 0x3368(%rip),%xmm0,%xmm2        # 60e0 <_IO_stdin_used+0xe0>
    2d77:	00 
    2d78:	c5 f8 28 ec          	vmovaps %xmm4,%xmm5
    2d7c:	c5 fb 11 54 24 10    	vmovsd %xmm2,0x10(%rsp)
    2d82:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2d88:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
    2d8c:	c5 fa 11 6c 24 0c    	vmovss %xmm5,0xc(%rsp)
    2d92:	c4 c1 43 2a ce       	vcvtsi2sd %r14d,%xmm7,%xmm1
    2d97:	c5 f3 59 4c 24 10    	vmulsd 0x10(%rsp),%xmm1,%xmm1
    2d9d:	c5 fa 11 64 24 08    	vmovss %xmm4,0x8(%rsp)
    2da3:	c5 f3 5e 4c 24 18    	vdivsd 0x18(%rsp),%xmm1,%xmm1
    2da9:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
            std::complex<float> w = std::polar(1.0f, -angle);
    2dad:	c5 fa 11 4c 24 04    	vmovss %xmm1,0x4(%rsp)
    2db3:	c5 f0 57 05 35 33 00 	vxorps 0x3335(%rip),%xmm1,%xmm0        # 60f0 <_IO_stdin_used+0xf0>
    2dba:	00 
  using ::sin;

#ifndef __CORRECT_ISO_CPP_MATH_H_PROTO
  inline _GLIBCXX_CONSTEXPR float
  sin(float __x)
  { return __builtin_sinf(__x); }
    2dbb:	e8 60 f3 ff ff       	call   2120 <sinf@plt>
  { return __builtin_cosf(__x); }
    2dc0:	c5 fa 10 4c 24 04    	vmovss 0x4(%rsp),%xmm1
  { return __builtin_sinf(__x); }
    2dc6:	c5 fa 11 04 24       	vmovss %xmm0,(%rsp)
  { return __builtin_cosf(__x); }
    2dcb:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    2dcf:	e8 2c f3 ff ff       	call   2100 <cosf@plt>
      template<class _Tp>
        _GLIBCXX20_CONSTEXPR complex&
        operator*=(const complex<_Tp>& __z)
	{
	  const _ComplexT __t = __z.__rep();
	  _M_value *= __t;
    2dd4:	c5 fa 10 1c 24       	vmovss (%rsp),%xmm3
    2dd9:	c4 a1 7a 10 4c f3 04 	vmovss 0x4(%rbx,%r14,8),%xmm1
    2de0:	c4 a1 7a 10 3c f3    	vmovss (%rbx,%r14,8),%xmm7
        operator*=(const complex<_Tp>& __z)
    2de6:	c5 fa 10 64 24 08    	vmovss 0x8(%rsp),%xmm4
	  _M_value *= __t;
    2dec:	c5 f2 59 d0          	vmulss %xmm0,%xmm1,%xmm2
    2df0:	c5 fa 10 6c 24 0c    	vmovss 0xc(%rsp),%xmm5
    2df6:	c5 f2 59 f3          	vmulss %xmm3,%xmm1,%xmm6
    2dfa:	c4 e2 41 b9 d3       	vfmadd231ss %xmm3,%xmm7,%xmm2
    2dff:	c4 e2 41 bb f0       	vfmsub231ss %xmm0,%xmm7,%xmm6
    2e04:	c5 f8 2e d6          	vucomiss %xmm6,%xmm2
    2e08:	7a 62                	jp     2e6c <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x14c>
        operator+=(const complex<_Tp>& __z)
    2e0a:	49 ff c6             	inc    %r14
	  _M_value += __z.__rep();
    2e0d:	c5 d2 58 ee          	vaddss %xmm6,%xmm5,%xmm5
    2e11:	c5 da 58 e2          	vaddss %xmm2,%xmm4,%xmm4
        for (int n = 0; n < N; n++)
    2e15:	4c 39 f5             	cmp    %r14,%rbp
    2e18:	0f 85 6a ff ff ff    	jne    2d88 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x68>
        out[k] = sum;
    2e1e:	c4 81 7a 11 6c e5 00 	vmovss %xmm5,0x0(%r13,%r12,8)
    2e25:	c4 81 7a 11 64 e5 04 	vmovss %xmm4,0x4(%r13,%r12,8)
    for (int k = 0; k < N; k++)
    2e2c:	49 ff c4             	inc    %r12
    2e2f:	4c 39 e5             	cmp    %r12,%rbp
    2e32:	0f 85 28 ff ff ff    	jne    2d60 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
}
    2e38:	48 83 c4 20          	add    $0x20,%rsp
    2e3c:	5b                   	pop    %rbx
    2e3d:	5d                   	pop    %rbp
    2e3e:	41 5c                	pop    %r12
    2e40:	41 5d                	pop    %r13
    2e42:	41 5e                	pop    %r14
    2e44:	c3                   	ret
            float angle = 2.0f * M_PI * k * n / N;
    2e45:	48 89 f8             	mov    %rdi,%rax
    2e48:	48 89 fa             	mov    %rdi,%rdx
    2e4b:	c5 e1 57 db          	vxorpd %xmm3,%xmm3,%xmm3
    2e4f:	48 d1 e8             	shr    $1,%rax
    2e52:	83 e2 01             	and    $0x1,%edx
    2e55:	48 09 d0             	or     %rdx,%rax
    2e58:	c4 e1 e3 2a c0       	vcvtsi2sd %rax,%xmm3,%xmm0
    2e5d:	c5 fb 58 d8          	vaddsd %xmm0,%xmm0,%xmm3
    2e61:	c5 fb 11 5c 24 18    	vmovsd %xmm3,0x18(%rsp)
    2e67:	e9 e7 fe ff ff       	jmp    2d53 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x33>
    2e6c:	c5 f8 28 d0          	vmovaps %xmm0,%xmm2
    2e70:	c5 f8 28 c7          	vmovaps %xmm7,%xmm0
    2e74:	c5 fa 11 6c 24 04    	vmovss %xmm5,0x4(%rsp)
    2e7a:	49 ff c6             	inc    %r14
    2e7d:	c5 fa 11 24 24       	vmovss %xmm4,(%rsp)
    2e82:	e8 f9 f3 ff ff       	call   2280 <__mulsc3@plt>
    2e87:	c5 fa 10 6c 24 04    	vmovss 0x4(%rsp),%xmm5
    2e8d:	c5 fa 10 24 24       	vmovss (%rsp),%xmm4
    2e92:	c4 e1 f9 7e c0       	vmovq  %xmm0,%rax
    2e97:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    2e9b:	48 c1 e8 20          	shr    $0x20,%rax
    2e9f:	c5 d2 58 e9          	vaddss %xmm1,%xmm5,%xmm5
    2ea3:	c5 f9 6e c0          	vmovd  %eax,%xmm0
    2ea7:	c5 da 58 e0          	vaddss %xmm0,%xmm4,%xmm4
        for (int n = 0; n < N; n++)
    2eab:	4c 39 f5             	cmp    %r14,%rbp
    2eae:	0f 85 d4 fe ff ff    	jne    2d88 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x68>
    2eb4:	e9 65 ff ff ff       	jmp    2e1e <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0xfe>
    2eb9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002ec0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_>:
{
    2ec0:	41 57                	push   %r15
    2ec2:	41 56                	push   %r14
    2ec4:	41 55                	push   %r13
    2ec6:	41 54                	push   %r12
    2ec8:	55                   	push   %rbp
    2ec9:	53                   	push   %rbx
    2eca:	48 83 ec 58          	sub    $0x58,%rsp
    for (size_t i = 0; i < N; i++)
    2ece:	48 85 ff             	test   %rdi,%rdi
    2ed1:	0f 84 1a 02 00 00    	je     30f1 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x231>
    2ed7:	48 89 fd             	mov    %rdi,%rbp
    2eda:	49 89 f4             	mov    %rsi,%r12
    2edd:	49 89 d5             	mov    %rdx,%r13
        out[i] = data[bit_reverse(i, log2(N))];
    2ee0:	0f 88 21 02 00 00    	js     3107 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x247>
    2ee6:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    2eea:	c4 e1 cb 2a c7       	vcvtsi2sd %rdi,%xmm6,%xmm0
    2eef:	c4 e1 f9 7e c3       	vmovq  %xmm0,%rbx
    for (size_t i = 0; i < N; i++)
    2ef4:	45 31 ff             	xor    %r15d,%r15d
    2ef7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2efe:	00 00 
        out[i] = data[bit_reverse(i, log2(N))];
    2f00:	c4 e1 f9 6e c3       	vmovq  %rbx,%xmm0
    2f05:	e8 46 f3 ff ff       	call   2250 <log2@plt>
    2f0a:	c5 fb 2c f8          	vcvttsd2si %xmm0,%edi
    for (int i = 0; i < log2n; i++)
    2f0e:	85 ff                	test   %edi,%edi
    2f10:	0f 8e ea 01 00 00    	jle    3100 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x240>
    2f16:	4c 89 f9             	mov    %r15,%rcx
    2f19:	31 d2                	xor    %edx,%edx
    size_t result = 0;
    2f1b:	31 c0                	xor    %eax,%eax
    2f1d:	0f 1f 00             	nopl   (%rax)
        result |= (x & 1);
    2f20:	48 89 ce             	mov    %rcx,%rsi
        result <<= 1;
    2f23:	48 01 c0             	add    %rax,%rax
    for (int i = 0; i < log2n; i++)
    2f26:	ff c2                	inc    %edx
        x >>= 1;
    2f28:	48 d1 e9             	shr    $1,%rcx
        result |= (x & 1);
    2f2b:	83 e6 01             	and    $0x1,%esi
    2f2e:	48 09 f0             	or     %rsi,%rax
    for (int i = 0; i < log2n; i++)
    2f31:	39 d7                	cmp    %edx,%edi
    2f33:	75 eb                	jne    2f20 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x60>
    2f35:	48 c1 e0 03          	shl    $0x3,%rax
        out[i] = data[bit_reverse(i, log2(N))];
    2f39:	49 8b 14 24          	mov    (%r12),%rdx
    2f3d:	4d 8b 45 00          	mov    0x0(%r13),%r8
    2f41:	c5 fa 10 04 02       	vmovss (%rdx,%rax,1),%xmm0
    2f46:	c4 81 7a 11 04 f8    	vmovss %xmm0,(%r8,%r15,8)
    2f4c:	c5 fa 10 44 02 04    	vmovss 0x4(%rdx,%rax,1),%xmm0
    2f52:	c4 81 7a 11 44 f8 04 	vmovss %xmm0,0x4(%r8,%r15,8)
    for (size_t i = 0; i < N; i++)
    2f59:	49 ff c7             	inc    %r15
    2f5c:	4c 39 fd             	cmp    %r15,%rbp
    2f5f:	75 9f                	jne    2f00 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
    for (int len = 2; len <= N; len <<= 1)
    2f61:	49 83 ff 01          	cmp    $0x1,%r15
    2f65:	0f 84 86 01 00 00    	je     30f1 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x231>
    2f6b:	c5 fa 10 2d 91 30 00 	vmovss 0x3091(%rip),%xmm5        # 6004 <_IO_stdin_used+0x4>
    2f72:	00 
    2f73:	c5 fa 10 25 8d 30 00 	vmovss 0x308d(%rip),%xmm4        # 6008 <_IO_stdin_used+0x8>
    2f7a:	00 
    2f7b:	bf 02 00 00 00       	mov    $0x2,%edi
    2f80:	c5 fa 10 35 84 30 00 	vmovss 0x3084(%rip),%xmm6        # 600c <_IO_stdin_used+0xc>
    2f87:	00 
    2f88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2f8f:	00 
        int half = len / 2;
    2f90:	41 89 fd             	mov    %edi,%r13d
    2f93:	41 d1 fd             	sar    $1,%r13d
        for (int start = 0; start < N; start += len)
    2f96:	83 ff 01             	cmp    $0x1,%edi
    2f99:	0f 8e e1 00 00 00    	jle    3080 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1c0>
    2f9f:	48 63 d7             	movslq %edi,%rdx
    2fa2:	49 63 cd             	movslq %r13d,%rcx
    2fa5:	4c 89 c6             	mov    %r8,%rsi
    2fa8:	48 8d 04 d5 00 00 00 	lea    0x0(,%rdx,8),%rax
    2faf:	00 
    2fb0:	49 8d 0c c8          	lea    (%r8,%rcx,8),%rcx
        int half = len / 2;
    2fb4:	49 89 d6             	mov    %rdx,%r14
    2fb7:	48 29 c6             	sub    %rax,%rsi
    2fba:	48 29 c1             	sub    %rax,%rcx
    2fbd:	0f 1f 00             	nopl   (%rax)
            for (int i = 0; i < half; i++)
    2fc0:	4a 8d 1c f5 00 00 00 	lea    0x0(,%r14,8),%rbx
    2fc7:	00 
        for (int start = 0; start < N; start += len)
    2fc8:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    2fcc:	c5 f8 28 d6          	vmovaps %xmm6,%xmm2
            for (int i = 0; i < half; i++)
    2fd0:	45 31 e4             	xor    %r12d,%r12d
    2fd3:	48 8d 2c 33          	lea    (%rbx,%rsi,1),%rbp
    2fd7:	48 01 cb             	add    %rcx,%rbx
    2fda:	eb 0c                	jmp    2fe8 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x128>
    2fdc:	0f 1f 40 00          	nopl   0x0(%rax)
    2fe0:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
    2fe4:	c5 f8 28 d1          	vmovaps %xmm1,%xmm2
                std::complex<float> a = out[start + i];
    2fe8:	c5 fa 10 4b 04       	vmovss 0x4(%rbx),%xmm1
    2fed:	c5 7a 10 13          	vmovss (%rbx),%xmm10
    2ff1:	c5 7a 10 45 00       	vmovss 0x0(%rbp),%xmm8
    2ff6:	c5 fa 10 7d 04       	vmovss 0x4(%rbp),%xmm7
	  _M_value *= __t;
    2ffb:	c5 f2 59 c2          	vmulss %xmm2,%xmm1,%xmm0
    2fff:	c5 72 59 cb          	vmulss %xmm3,%xmm1,%xmm9
    3003:	c4 e2 29 b9 c3       	vfmadd231ss %xmm3,%xmm10,%xmm0
    3008:	c4 62 29 bb ca       	vfmsub231ss %xmm2,%xmm10,%xmm9
    300d:	c4 c1 78 2e c1       	vucomiss %xmm9,%xmm0
    3012:	0f 8a 86 01 00 00    	jp     319e <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x2de>
	  _M_value += __z.__rep();
    3018:	c4 c1 32 58 c8       	vaddss %xmm8,%xmm9,%xmm1
	  _M_value -= __z.__rep();
    301d:	c4 41 3a 5c c1       	vsubss %xmm9,%xmm8,%xmm8
                out[start + i] = a + w * b;
    3022:	c5 fa 11 4d 00       	vmovss %xmm1,0x0(%rbp)
	  _M_value += __z.__rep();
    3027:	c5 fa 58 cf          	vaddss %xmm7,%xmm0,%xmm1
	  _M_value -= __z.__rep();
    302b:	c5 c2 5c f8          	vsubss %xmm0,%xmm7,%xmm7
	  _M_value *= __t;
    302f:	c5 d2 59 c2          	vmulss %xmm2,%xmm5,%xmm0
    3033:	c5 fa 11 4d 04       	vmovss %xmm1,0x4(%rbp)
    3038:	c5 d2 59 cb          	vmulss %xmm3,%xmm5,%xmm1
                out[start + i + half] = a - w * b;
    303c:	c5 7a 11 03          	vmovss %xmm8,(%rbx)
    3040:	c4 e2 59 b9 c3       	vfmadd231ss %xmm3,%xmm4,%xmm0
    3045:	c5 fa 11 7b 04       	vmovss %xmm7,0x4(%rbx)
    304a:	c4 e2 59 bb ca       	vfmsub231ss %xmm2,%xmm4,%xmm1
    304f:	c5 f8 2e c1          	vucomiss %xmm1,%xmm0
    3053:	0f 8a d4 00 00 00    	jp     312d <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x26d>
            for (int i = 0; i < half; i++)
    3059:	41 ff c4             	inc    %r12d
    305c:	48 83 c5 08          	add    $0x8,%rbp
    3060:	48 83 c3 08          	add    $0x8,%rbx
    3064:	45 39 e5             	cmp    %r12d,%r13d
    3067:	0f 8f 73 ff ff ff    	jg     2fe0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x120>
        for (int start = 0; start < N; start += len)
    306d:	4a 8d 04 32          	lea    (%rdx,%r14,1),%rax
    3071:	4d 39 fe             	cmp    %r15,%r14
    3074:	73 0a                	jae    3080 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1c0>
    3076:	49 89 c6             	mov    %rax,%r14
    3079:	e9 42 ff ff ff       	jmp    2fc0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x100>
    307e:	66 90                	xchg   %ax,%ax
    for (int len = 2; len <= N; len <<= 1)
    3080:	01 ff                	add    %edi,%edi
    3082:	4c 89 04 24          	mov    %r8,(%rsp)
    3086:	48 63 c7             	movslq %edi,%rax
    3089:	49 39 c7             	cmp    %rax,%r15
    308c:	72 63                	jb     30f1 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x231>
        float angle = 2.0f * M_PI / len;
    308e:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    3092:	89 7c 24 10          	mov    %edi,0x10(%rsp)
    3096:	c5 cb 2a c7          	vcvtsi2sd %edi,%xmm6,%xmm0
    309a:	c5 fb 10 35 3e 30 00 	vmovsd 0x303e(%rip),%xmm6        # 60e0 <_IO_stdin_used+0xe0>
    30a1:	00 
    30a2:	c5 cb 5e c0          	vdivsd %xmm0,%xmm6,%xmm0
    30a6:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
    30aa:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    30ae:	c5 fa 11 4c 24 30    	vmovss %xmm1,0x30(%rsp)
    30b4:	e8 47 f0 ff ff       	call   2100 <cosf@plt>
        std::complex<float> w_step = std::polar(1.0f, -angle);
    30b9:	c5 fa 10 4c 24 30    	vmovss 0x30(%rsp),%xmm1
    30bf:	c5 fa 11 44 24 0c    	vmovss %xmm0,0xc(%rsp)
    30c5:	c5 f0 57 05 23 30 00 	vxorps 0x3023(%rip),%xmm1,%xmm0        # 60f0 <_IO_stdin_used+0xf0>
    30cc:	00 
  { return __builtin_sinf(__x); }
    30cd:	e8 4e f0 ff ff       	call   2120 <sinf@plt>
    30d2:	c5 fa 10 64 24 0c    	vmovss 0xc(%rsp),%xmm4
    30d8:	8b 7c 24 10          	mov    0x10(%rsp),%edi
    30dc:	4c 8b 04 24          	mov    (%rsp),%r8
    30e0:	c5 fa 10 35 24 2f 00 	vmovss 0x2f24(%rip),%xmm6        # 600c <_IO_stdin_used+0xc>
    30e7:	00 
    30e8:	c5 f8 28 e8          	vmovaps %xmm0,%xmm5
    30ec:	e9 9f fe ff ff       	jmp    2f90 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0xd0>
}
    30f1:	48 83 c4 58          	add    $0x58,%rsp
    30f5:	5b                   	pop    %rbx
    30f6:	5d                   	pop    %rbp
    30f7:	41 5c                	pop    %r12
    30f9:	41 5d                	pop    %r13
    30fb:	41 5e                	pop    %r14
    30fd:	41 5f                	pop    %r15
    30ff:	c3                   	ret
    for (int i = 0; i < log2n; i++)
    3100:	31 c0                	xor    %eax,%eax
    3102:	e9 32 fe ff ff       	jmp    2f39 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x79>
        out[i] = data[bit_reverse(i, log2(N))];
    3107:	48 89 f8             	mov    %rdi,%rax
    310a:	48 89 fa             	mov    %rdi,%rdx
    310d:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    3111:	48 d1 e8             	shr    $1,%rax
    3114:	83 e2 01             	and    $0x1,%edx
    3117:	48 09 d0             	or     %rdx,%rax
    311a:	c4 e1 cb 2a c0       	vcvtsi2sd %rax,%xmm6,%xmm0
    311f:	c5 fb 58 f0          	vaddsd %xmm0,%xmm0,%xmm6
    3123:	c4 e1 f9 7e f3       	vmovq  %xmm6,%rbx
    3128:	e9 c7 fd ff ff       	jmp    2ef4 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x34>
    312d:	c5 f8 28 c4          	vmovaps %xmm4,%xmm0
    3131:	c5 f8 28 cd          	vmovaps %xmm5,%xmm1
    3135:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
    313a:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
    313f:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
    3144:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
    3149:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    314d:	c5 fa 11 6c 24 0c    	vmovss %xmm5,0xc(%rsp)
    3153:	c5 fa 11 24 24       	vmovss %xmm4,(%rsp)
    3158:	e8 23 f1 ff ff       	call   2280 <__mulsc3@plt>
    315d:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    3162:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    3167:	c5 f9 6f f0          	vmovdqa %xmm0,%xmm6
    316b:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    316f:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    3174:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    3179:	c5 c8 c6 f6 55       	vshufps $0x55,%xmm6,%xmm6,%xmm6
    317e:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    3182:	c5 f9 6f c6          	vmovdqa %xmm6,%xmm0
    3186:	c5 fa 10 6c 24 0c    	vmovss 0xc(%rsp),%xmm5
    318c:	c5 fa 10 35 78 2e 00 	vmovss 0x2e78(%rip),%xmm6        # 600c <_IO_stdin_used+0xc>
    3193:	00 
    3194:	c5 fa 10 24 24       	vmovss (%rsp),%xmm4
    3199:	e9 bb fe ff ff       	jmp    3059 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x199>
    319e:	c5 78 29 d0          	vmovaps %xmm10,%xmm0
    31a2:	48 89 54 24 48       	mov    %rdx,0x48(%rsp)
    31a7:	48 89 74 24 40       	mov    %rsi,0x40(%rsp)
    31ac:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
    31b1:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
    31b6:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    31ba:	c5 fa 11 6c 24 34    	vmovss %xmm5,0x34(%rsp)
    31c0:	c5 fa 11 64 24 28    	vmovss %xmm4,0x28(%rsp)
    31c6:	c5 7a 11 44 24 20    	vmovss %xmm8,0x20(%rsp)
    31cc:	c5 fa 11 7c 24 18    	vmovss %xmm7,0x18(%rsp)
    31d2:	c5 fa 11 5c 24 0c    	vmovss %xmm3,0xc(%rsp)
    31d8:	c5 fa 11 14 24       	vmovss %xmm2,(%rsp)
    31dd:	e8 9e f0 ff ff       	call   2280 <__mulsc3@plt>
    31e2:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
    31e7:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    31ec:	c5 f9 6f f0          	vmovdqa %xmm0,%xmm6
    31f0:	c5 79 6f c8          	vmovdqa %xmm0,%xmm9
    31f4:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    31f8:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    31fd:	c5 c8 c6 f6 55       	vshufps $0x55,%xmm6,%xmm6,%xmm6
    3202:	c5 fa 10 6c 24 34    	vmovss 0x34(%rsp),%xmm5
    3208:	c5 f9 6f c6          	vmovdqa %xmm6,%xmm0
    320c:	c5 fa 10 64 24 28    	vmovss 0x28(%rsp),%xmm4
    3212:	c5 fa 10 35 f2 2d 00 	vmovss 0x2df2(%rip),%xmm6        # 600c <_IO_stdin_used+0xc>
    3219:	00 
    321a:	c5 7a 10 44 24 20    	vmovss 0x20(%rsp),%xmm8
    3220:	c5 fa 10 7c 24 18    	vmovss 0x18(%rsp),%xmm7
    3226:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    322b:	c5 fa 10 5c 24 0c    	vmovss 0xc(%rsp),%xmm3
    3231:	c5 fa 10 14 24       	vmovss (%rsp),%xmm2
    3236:	e9 dd fd ff ff       	jmp    3018 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x158>
    323b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000003240 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_>:
{
    3240:	41 57                	push   %r15
    3242:	41 56                	push   %r14
    3244:	41 55                	push   %r13
    3246:	41 54                	push   %r12
    3248:	55                   	push   %rbp
    const int half = N / 2;
    3249:	48 89 fd             	mov    %rdi,%rbp
{
    324c:	53                   	push   %rbx
    const int half = N / 2;
    324d:	48 d1 ed             	shr    $1,%rbp
{
    3250:	48 83 ec 78          	sub    $0x78,%rsp
      { return _M_data_ptr(this->_M_impl._M_start); }
    3254:	48 8b 36             	mov    (%rsi),%rsi
    3257:	4c 8b 22             	mov    (%rdx),%r12
    for (int stride = half; stride >= 1; stride >>= 1)
    325a:	85 ed                	test   %ebp,%ebp
    325c:	0f 8e 93 02 00 00    	jle    34f5 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x2b5>
    3262:	4c 63 ed             	movslq %ebp,%r13
        float angle = block_size * M_PI / N;
    3265:	48 85 ff             	test   %rdi,%rdi
    3268:	0f 88 ab 01 00 00    	js     3419 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1d9>
    326e:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
    3272:	c4 e1 eb 2a f7       	vcvtsi2sd %rdi,%xmm2,%xmm6
    3277:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    auto *p2 = out.data();
    327c:	4c 89 e3             	mov    %r12,%rbx
    327f:	41 89 ee             	mov    %ebp,%r14d
    3282:	4c 89 64 24 20       	mov    %r12,0x20(%rsp)
    3287:	49 89 f4             	mov    %rsi,%r12
    328a:	c5 fb 11 74 24 18    	vmovsd %xmm6,0x18(%rsp)
        float angle = block_size * M_PI / N;
    3290:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
        int block_size = stride << 1;
    3294:	43 8d 2c 36          	lea    (%r14,%r14,1),%ebp
        float angle = block_size * M_PI / N;
    3298:	c5 eb 2a c5          	vcvtsi2sd %ebp,%xmm2,%xmm0
    329c:	c5 fb 59 0d 44 2e 00 	vmulsd 0x2e44(%rip),%xmm0,%xmm1        # 60e8 <_IO_stdin_used+0xe8>
    32a3:	00 
    32a4:	c5 f3 5e 4c 24 18    	vdivsd 0x18(%rsp),%xmm1,%xmm1
    32aa:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
        std::complex<float> w_step = std::polar(1.0f, -angle);
    32ae:	c5 fa 11 4c 24 08    	vmovss %xmm1,0x8(%rsp)
    32b4:	c5 f0 57 05 34 2e 00 	vxorps 0x2e34(%rip),%xmm1,%xmm0        # 60f0 <_IO_stdin_used+0xf0>
    32bb:	00 
    32bc:	e8 5f ee ff ff       	call   2120 <sinf@plt>
  { return __builtin_cosf(__x); }
    32c1:	c5 fa 10 4c 24 08    	vmovss 0x8(%rsp),%xmm1
  { return __builtin_sinf(__x); }
    32c7:	c5 fa 11 44 24 0c    	vmovss %xmm0,0xc(%rsp)
  { return __builtin_cosf(__x); }
    32cd:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    32d1:	e8 2a ee ff ff       	call   2100 <cosf@plt>
        int block_count = N / block_size;
    32d6:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    32db:	48 63 f5             	movslq %ebp,%rsi
    32de:	31 d2                	xor    %edx,%edx
    32e0:	48 f7 f6             	div    %rsi
    32e3:	41 89 c0             	mov    %eax,%r8d
        for (int b = 0; b < block_count; b++)
    32e6:	85 c0                	test   %eax,%eax
    32e8:	0f 8e e2 00 00 00    	jle    33d0 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x190>
    32ee:	c5 fa 10 64 24 0c    	vmovss 0xc(%rsp),%xmm4
    32f4:	4d 63 fe             	movslq %r14d,%r15
    32f7:	48 c1 e6 03          	shl    $0x3,%rsi
    32fb:	4c 89 e2             	mov    %r12,%rdx
    32fe:	4a 8d 0c fd 00 00 00 	lea    0x0(,%r15,8),%rcx
    3305:	00 
    3306:	c5 fa 10 15 fe 2c 00 	vmovss 0x2cfe(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    330d:	00 
    330e:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    3312:	31 c0                	xor    %eax,%eax
    3314:	48 8d 2c 0b          	lea    (%rbx,%rcx,1),%rbp
    3318:	c5 78 28 c4          	vmovaps %xmm4,%xmm8
    331c:	c5 78 28 c8          	vmovaps %xmm0,%xmm9
            for (int w_count = 0; w_count < stride; w_count++)
    3320:	48 89 ef             	mov    %rbp,%rdi
    for (int stride = half; stride >= 1; stride >>= 1)
    3323:	49 89 d1             	mov    %rdx,%r9
    3326:	48 29 cf             	sub    %rcx,%rdi
    3329:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
                std::complex<float> a = p1[b * block_size + w_count];
    3330:	c4 81 7a 10 4c f9 04 	vmovss 0x4(%r9,%r15,8),%xmm1
    3337:	c4 81 7a 10 3c f9    	vmovss (%r9,%r15,8),%xmm7
    333d:	c4 c1 7a 10 31       	vmovss (%r9),%xmm6
    3342:	c4 c1 7a 10 69 04    	vmovss 0x4(%r9),%xmm5
    3348:	c5 ea 59 c1          	vmulss %xmm1,%xmm2,%xmm0
    334c:	c5 e2 59 e1          	vmulss %xmm1,%xmm3,%xmm4
    3350:	c4 e2 61 b9 c7       	vfmadd231ss %xmm7,%xmm3,%xmm0
    3355:	c4 e2 69 bb e7       	vfmsub231ss %xmm7,%xmm2,%xmm4
    335a:	c5 f8 2e c4          	vucomiss %xmm4,%xmm0
    335e:	0f 8a e5 00 00 00    	jp     3449 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x209>
	  _M_value += __z.__rep();
    3364:	c5 da 58 ce          	vaddss %xmm6,%xmm4,%xmm1
	  _M_value -= __z.__rep();
    3368:	c5 ca 5c e4          	vsubss %xmm4,%xmm6,%xmm4
            for (int w_count = 0; w_count < stride; w_count++)
    336c:	49 83 c1 08          	add    $0x8,%r9
                p2[b * stride + w_count] = a + wb;
    3370:	c5 fa 11 0f          	vmovss %xmm1,(%rdi)
	  _M_value += __z.__rep();
    3374:	c5 fa 58 cd          	vaddss %xmm5,%xmm0,%xmm1
	  _M_value -= __z.__rep();
    3378:	c5 d2 5c c0          	vsubss %xmm0,%xmm5,%xmm0
    337c:	c5 fa 11 4f 04       	vmovss %xmm1,0x4(%rdi)
                p2[b * stride + w_count + half] = a - wb;
    3381:	c4 a1 7a 11 24 ef    	vmovss %xmm4,(%rdi,%r13,8)
    3387:	c4 a1 7a 11 44 ef 04 	vmovss %xmm0,0x4(%rdi,%r13,8)
            for (int w_count = 0; w_count < stride; w_count++)
    338e:	48 83 c7 08          	add    $0x8,%rdi
    3392:	48 39 fd             	cmp    %rdi,%rbp
    3395:	75 99                	jne    3330 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0xf0>
	  _M_value *= __t;
    3397:	c5 ba 59 c2          	vmulss %xmm2,%xmm8,%xmm0
    339b:	c4 c1 62 59 c8       	vmulss %xmm8,%xmm3,%xmm1
    33a0:	c4 c2 61 b9 c1       	vfmadd231ss %xmm9,%xmm3,%xmm0
    33a5:	c4 e2 31 bb ca       	vfmsub231ss %xmm2,%xmm9,%xmm1
    33aa:	c5 f8 2e c8          	vucomiss %xmm0,%xmm1
    33ae:	0f 8a 49 01 00 00    	jp     34fd <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x2bd>
        for (int b = 0; b < block_count; b++)
    33b4:	ff c0                	inc    %eax
    33b6:	48 01 cd             	add    %rcx,%rbp
    33b9:	48 01 f2             	add    %rsi,%rdx
    33bc:	41 39 c0             	cmp    %eax,%r8d
    33bf:	74 0f                	je     33d0 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x190>
    33c1:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
    33c5:	c5 f8 28 d1          	vmovaps %xmm1,%xmm2
    33c9:	e9 52 ff ff ff       	jmp    3320 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0xe0>
    33ce:	66 90                	xchg   %ax,%ax
    for (int stride = half; stride >= 1; stride >>= 1)
    33d0:	41 d1 fe             	sar    $1,%r14d
    33d3:	74 0e                	je     33e3 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1a3>
    33d5:	4c 89 e0             	mov    %r12,%rax
    33d8:	49 89 dc             	mov    %rbx,%r12
    33db:	48 89 c3             	mov    %rax,%rbx
    33de:	e9 ad fe ff ff       	jmp    3290 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x50>
    33e3:	4c 8b 64 24 20       	mov    0x20(%rsp),%r12
    33e8:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    if (p1 != out.data())
    33ed:	49 39 dc             	cmp    %rbx,%r12
    33f0:	74 48                	je     343a <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1fa>
        std::copy(p1, p1 + N, out.data());
    33f2:	48 8d 14 fd 00 00 00 	lea    0x0(,%rdi,8),%rdx
    33f9:	00 
	_GLIBCXX20_CONSTEXPR
	static _Up*
	__copy_m(_Tp* __first, _Tp* __last, _Up* __result)
	{
	  const ptrdiff_t _Num = __last - __first;
	  if (__builtin_expect(_Num > 1, true))
    33fa:	48 83 fa 08          	cmp    $0x8,%rdx
    33fe:	7e 34                	jle    3434 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1f4>
}
    3400:	48 83 c4 78          	add    $0x78,%rsp
	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
    3404:	48 89 de             	mov    %rbx,%rsi
    3407:	4c 89 e7             	mov    %r12,%rdi
    340a:	5b                   	pop    %rbx
    340b:	5d                   	pop    %rbp
    340c:	41 5c                	pop    %r12
    340e:	41 5d                	pop    %r13
    3410:	41 5e                	pop    %r14
    3412:	41 5f                	pop    %r15
    3414:	e9 f7 ed ff ff       	jmp    2210 <memmove@plt>
        float angle = block_size * M_PI / N;
    3419:	48 89 f8             	mov    %rdi,%rax
    341c:	c5 e1 57 db          	vxorpd %xmm3,%xmm3,%xmm3
    3420:	83 e0 01             	and    $0x1,%eax
    3423:	48 09 e8             	or     %rbp,%rax
    3426:	c4 e1 e3 2a f0       	vcvtsi2sd %rax,%xmm3,%xmm6
    342b:	c5 cb 58 f6          	vaddsd %xmm6,%xmm6,%xmm6
    342f:	e9 43 fe ff ff       	jmp    3277 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x37>
	  else if (_Num == 1)
    3434:	0f 84 2e 01 00 00    	je     3568 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x328>
}
    343a:	48 83 c4 78          	add    $0x78,%rsp
    343e:	5b                   	pop    %rbx
    343f:	5d                   	pop    %rbp
    3440:	41 5c                	pop    %r12
    3442:	41 5d                	pop    %r13
    3444:	41 5e                	pop    %r14
    3446:	41 5f                	pop    %r15
    3448:	c3                   	ret
    3449:	c5 f8 28 c7          	vmovaps %xmm7,%xmm0
    344d:	48 89 74 24 68       	mov    %rsi,0x68(%rsp)
    3452:	4c 89 4c 24 60       	mov    %r9,0x60(%rsp)
    3457:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
    345c:	89 44 24 54          	mov    %eax,0x54(%rsp)
    3460:	44 89 44 24 50       	mov    %r8d,0x50(%rsp)
    3465:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
    346a:	48 89 7c 24 40       	mov    %rdi,0x40(%rsp)
    346f:	c5 7a 11 4c 24 38    	vmovss %xmm9,0x38(%rsp)
    3475:	c5 7a 11 44 24 34    	vmovss %xmm8,0x34(%rsp)
    347b:	c5 fa 11 74 24 30    	vmovss %xmm6,0x30(%rsp)
    3481:	c5 fa 11 6c 24 28    	vmovss %xmm5,0x28(%rsp)
    3487:	c5 fa 11 5c 24 0c    	vmovss %xmm3,0xc(%rsp)
    348d:	c5 fa 11 54 24 08    	vmovss %xmm2,0x8(%rsp)
    3493:	e8 e8 ed ff ff       	call   2280 <__mulsc3@plt>
    3498:	48 8b 74 24 68       	mov    0x68(%rsp),%rsi
    349d:	4c 8b 4c 24 60       	mov    0x60(%rsp),%r9
    34a2:	c4 c1 f9 7e c2       	vmovq  %xmm0,%r10
    34a7:	c5 f9 6f e0          	vmovdqa %xmm0,%xmm4
    34ab:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
    34b0:	8b 44 24 54          	mov    0x54(%rsp),%eax
    34b4:	49 c1 ea 20          	shr    $0x20,%r10
    34b8:	44 8b 44 24 50       	mov    0x50(%rsp),%r8d
    34bd:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    34c2:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    34c7:	c5 7a 10 4c 24 38    	vmovss 0x38(%rsp),%xmm9
    34cd:	c4 c1 79 6e c2       	vmovd  %r10d,%xmm0
    34d2:	c5 7a 10 44 24 34    	vmovss 0x34(%rsp),%xmm8
    34d8:	c5 fa 10 74 24 30    	vmovss 0x30(%rsp),%xmm6
    34de:	c5 fa 10 6c 24 28    	vmovss 0x28(%rsp),%xmm5
    34e4:	c5 fa 10 5c 24 0c    	vmovss 0xc(%rsp),%xmm3
    34ea:	c5 fa 10 54 24 08    	vmovss 0x8(%rsp),%xmm2
    34f0:	e9 6f fe ff ff       	jmp    3364 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x124>
    auto *p1 = data.data();
    34f5:	48 89 f3             	mov    %rsi,%rbx
    34f8:	e9 f0 fe ff ff       	jmp    33ed <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1ad>
    34fd:	c5 78 29 c8          	vmovaps %xmm9,%xmm0
    3501:	c5 78 29 c1          	vmovaps %xmm8,%xmm1
    3505:	48 89 74 24 40       	mov    %rsi,0x40(%rsp)
    350a:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    350f:	89 44 24 34          	mov    %eax,0x34(%rsp)
    3513:	44 89 44 24 30       	mov    %r8d,0x30(%rsp)
    3518:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    351d:	c5 7a 11 44 24 0c    	vmovss %xmm8,0xc(%rsp)
    3523:	c5 7a 11 4c 24 08    	vmovss %xmm9,0x8(%rsp)
    3529:	e8 52 ed ff ff       	call   2280 <__mulsc3@plt>
    352e:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    3533:	48 8b 54 24 38       	mov    0x38(%rsp),%rdx
    3538:	c4 e1 f9 7e c7       	vmovq  %xmm0,%rdi
    353d:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    3541:	8b 44 24 34          	mov    0x34(%rsp),%eax
    3545:	44 8b 44 24 30       	mov    0x30(%rsp),%r8d
    354a:	48 c1 ef 20          	shr    $0x20,%rdi
    354e:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    3553:	c5 7a 10 44 24 0c    	vmovss 0xc(%rsp),%xmm8
    3559:	c5 7a 10 4c 24 08    	vmovss 0x8(%rsp),%xmm9
    355f:	c5 f9 6e c7          	vmovd  %edi,%xmm0
    3563:	e9 4c fe ff ff       	jmp    33b4 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x174>
	{ *__to = *__from; }
    3568:	c5 fa 10 03          	vmovss (%rbx),%xmm0
    356c:	c4 c1 7a 11 04 24    	vmovss %xmm0,(%r12)
    3572:	c5 fa 10 43 04       	vmovss 0x4(%rbx),%xmm0
    3577:	c4 c1 7a 11 44 24 04 	vmovss %xmm0,0x4(%r12)
}
    357e:	e9 b7 fe ff ff       	jmp    343a <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1fa>
    3583:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    358a:	00 00 00 00 
    358e:	66 90                	xchg   %ax,%ax

0000000000003590 <_Z4fftriibP9complex_tS0_>:
{
    3590:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    3595:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    3599:	49 89 cb             	mov    %rcx,%r11
    359c:	41 ff 72 f8          	push   -0x8(%r10)
    35a0:	55                   	push   %rbp
    35a1:	48 89 e5             	mov    %rsp,%rbp
    35a4:	41 57                	push   %r15
    35a6:	41 56                	push   %r14
    35a8:	4d 89 c6             	mov    %r8,%r14
    35ab:	41 55                	push   %r13
    35ad:	41 54                	push   %r12
    35af:	41 52                	push   %r10
    35b1:	53                   	push   %rbx
    35b2:	89 f3                	mov    %esi,%ebx
    35b4:	48 83 ec 60          	sub    $0x60,%rsp
    if (n == 1)
    35b8:	83 ff 01             	cmp    $0x1,%edi
    35bb:	75 53                	jne    3610 <_Z4fftriibP9complex_tS0_+0x80>
        if (eo)
    35bd:	84 d2                	test   %dl,%dl
    35bf:	74 30                	je     35f1 <_Z4fftriibP9complex_tS0_+0x61>
            for (int q = 0; q < s; q++)
    35c1:	85 f6                	test   %esi,%esi
    35c3:	7e 2c                	jle    35f1 <_Z4fftriibP9complex_tS0_+0x61>
    35c5:	48 63 ce             	movslq %esi,%rcx
    35c8:	31 c0                	xor    %eax,%eax
    35ca:	48 c1 e1 03          	shl    $0x3,%rcx
    35ce:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    35d5:	00 00 00 00 
    35d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
                x[q] = y[q];
    35e0:	49 8b 14 06          	mov    (%r14,%rax,1),%rdx
    35e4:	49 89 14 03          	mov    %rdx,(%r11,%rax,1)
            for (int q = 0; q < s; q++)
    35e8:	48 83 c0 08          	add    $0x8,%rax
    35ec:	48 39 c1             	cmp    %rax,%rcx
    35ef:	75 ef                	jne    35e0 <_Z4fftriibP9complex_tS0_+0x50>
}
    35f1:	48 83 c4 60          	add    $0x60,%rsp
    35f5:	5b                   	pop    %rbx
    35f6:	41 5a                	pop    %r10
    35f8:	41 5c                	pop    %r12
    35fa:	41 5d                	pop    %r13
    35fc:	41 5e                	pop    %r14
    35fe:	41 5f                	pop    %r15
    3600:	5d                   	pop    %rbp
    3601:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    3605:	c3                   	ret
    3606:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    360d:	00 00 00 
    const float angle = 2 * M_PI / n;
    3610:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    const int m = n / 2;
    3614:	89 f8                	mov    %edi,%eax
        fftr(n / 2, 2 * s, !eo, y, x);
    3616:	83 f2 01             	xor    $0x1,%edx
    3619:	49 89 c8             	mov    %rcx,%r8
    const float angle = 2 * M_PI / n;
    361c:	c5 fb 2a c7          	vcvtsi2sd %edi,%xmm0,%xmm0
    const int m = n / 2;
    3620:	c1 e8 1f             	shr    $0x1f,%eax
        fftr(n / 2, 2 * s, !eo, y, x);
    3623:	0f b6 d2             	movzbl %dl,%edx
    3626:	48 89 4d b8          	mov    %rcx,-0x48(%rbp)
    const float angle = 2 * M_PI / n;
    362a:	c5 fb 10 0d ae 2a 00 	vmovsd 0x2aae(%rip),%xmm1        # 60e0 <_IO_stdin_used+0xe0>
    3631:	00 
    const int m = n / 2;
    3632:	01 f8                	add    %edi,%eax
        fftr(n / 2, 2 * s, !eo, y, x);
    3634:	8d 34 36             	lea    (%rsi,%rsi,1),%esi
    3637:	4c 89 f1             	mov    %r14,%rcx
    const int m = n / 2;
    363a:	d1 f8                	sar    $1,%eax
    363c:	41 89 fd             	mov    %edi,%r13d
        fftr(n / 2, 2 * s, !eo, y, x);
    363f:	89 c7                	mov    %eax,%edi
    const int m = n / 2;
    3641:	89 45 c0             	mov    %eax,-0x40(%rbp)
    3644:	41 89 c7             	mov    %eax,%r15d
    const float angle = 2 * M_PI / n;
    3647:	c5 f3 5e c8          	vdivsd %xmm0,%xmm1,%xmm1
    364b:	c5 f3 5a f9          	vcvtsd2ss %xmm1,%xmm1,%xmm7
    364f:	c4 c1 79 7e fc       	vmovd  %xmm7,%r12d
        fftr(n / 2, 2 * s, !eo, y, x);
    3654:	e8 37 ff ff ff       	call   3590 <_Z4fftriibP9complex_tS0_>
    3659:	c4 c1 79 6e c4       	vmovd  %r12d,%xmm0
    365e:	e8 9d ea ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    3663:	c4 c1 79 6e f4       	vmovd  %r12d,%xmm6
    3668:	c5 fa 11 45 c8       	vmovss %xmm0,-0x38(%rbp)
    366d:	c5 c8 57 05 7b 2a 00 	vxorps 0x2a7b(%rip),%xmm6,%xmm0        # 60f0 <_IO_stdin_used+0xf0>
    3674:	00 
  { return __builtin_sinf(__x); }
    3675:	e8 a6 ea ff ff       	call   2120 <sinf@plt>
    367a:	c5 f8 28 f8          	vmovaps %xmm0,%xmm7
        for (int p = 0; p < m; p++)
    367e:	41 83 fd 01          	cmp    $0x1,%r13d
    3682:	0f 8e 69 ff ff ff    	jle    35f1 <_Z4fftriibP9complex_tS0_+0x61>
    3688:	8d 43 ff             	lea    -0x1(%rbx),%eax
    368b:	4c 8b 5d b8          	mov    -0x48(%rbp),%r11
    368f:	c5 fa 10 75 c8       	vmovss -0x38(%rbp),%xmm6
    3694:	45 31 e4             	xor    %r12d,%r12d
    3697:	89 c2                	mov    %eax,%edx
    3699:	89 45 b0             	mov    %eax,-0x50(%rbp)
    369c:	44 0f af fb          	imul   %ebx,%r15d
    36a0:	45 31 ed             	xor    %r13d,%r13d
    36a3:	d1 ea                	shr    $1,%edx
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    36a5:	c5 7a 10 35 5f 29 00 	vmovss 0x295f(%rip),%xmm14        # 600c <_IO_stdin_used+0xc>
    36ac:	00 
    36ad:	c4 41 10 57 ed       	vxorps %xmm13,%xmm13,%xmm13
            if (s == 1)
    36b2:	8d 42 01             	lea    0x1(%rdx),%eax
    36b5:	89 d1                	mov    %edx,%ecx
    36b7:	48 89 4d 88          	mov    %rcx,-0x78(%rbp)
    36bb:	89 c2                	mov    %eax,%edx
    36bd:	49 8d 4e 10          	lea    0x10(%r14),%rcx
    36c1:	48 89 4d 80          	mov    %rcx,-0x80(%rbp)
    36c5:	d1 ea                	shr    $1,%edx
    36c7:	89 c1                	mov    %eax,%ecx
    36c9:	83 e0 fe             	and    $0xfffffffe,%eax
    36cc:	48 c1 e2 05          	shl    $0x5,%rdx
    36d0:	83 e1 01             	and    $0x1,%ecx
    36d3:	44 89 7d c4          	mov    %r15d,-0x3c(%rbp)
        for (int p = 0; p < m; p++)
    36d7:	4d 89 f7             	mov    %r14,%r15
    36da:	48 89 55 98          	mov    %rdx,-0x68(%rbp)
    36de:	89 4d 94             	mov    %ecx,-0x6c(%rbp)
    36e1:	89 45 90             	mov    %eax,-0x70(%rbp)
            if (s == 1)
    36e4:	83 fb 01             	cmp    $0x1,%ebx
    36e7:	0f 84 eb 01 00 00    	je     38d8 <_Z4fftriibP9complex_tS0_+0x348>
    36ed:	0f 1f 00             	nopl   (%rax)
                for (int q = 0; q < s; q += 2)
    36f0:	85 db                	test   %ebx,%ebx
    36f2:	0f 8e aa 01 00 00    	jle    38a2 <_Z4fftriibP9complex_tS0_+0x312>
                    const complex_t b = y[q + s * (2 * p + 1)] * wp;
    36f8:	42 8d 14 63          	lea    (%rbx,%r12,2),%edx
    36fc:	49 63 cc             	movslq %r12d,%rcx
    36ff:	48 63 d2             	movslq %edx,%rdx
    3702:	48 8d 34 cd 00 00 00 	lea    0x0(,%rcx,8),%rsi
    3709:	00 
    370a:	48 8d 3c d5 10 00 00 	lea    0x10(,%rdx,8),%rdi
    3711:	00 
    3712:	48 89 55 a0          	mov    %rdx,-0x60(%rbp)
    3716:	48 63 55 c4          	movslq -0x3c(%rbp),%rdx
    371a:	49 8d 04 33          	lea    (%r11,%rsi,1),%rax
    371e:	4d 8d 14 3e          	lea    (%r14,%rdi,1),%r10
    3722:	48 89 7d c8          	mov    %rdi,-0x38(%rbp)
    3726:	48 8d 79 01          	lea    0x1(%rcx),%rdi
    372a:	4c 8d 04 d5 00 00 00 	lea    0x0(,%rdx,8),%r8
    3731:	00 
    3732:	48 c1 e7 04          	shl    $0x4,%rdi
    3736:	48 89 55 a8          	mov    %rdx,-0x58(%rbp)
    373a:	4b 8d 14 03          	lea    (%r11,%r8,1),%rdx
    373e:	4c 89 55 b8          	mov    %r10,-0x48(%rbp)
    3742:	4d 8d 14 3e          	lea    (%r14,%rdi,1),%r10
    3746:	49 89 d1             	mov    %rdx,%r9
    3749:	4d 29 d1             	sub    %r10,%r9
    374c:	49 83 c1 0c          	add    $0xc,%r9
    3750:	49 83 f9 18          	cmp    $0x18,%r9
    3754:	49 89 c1             	mov    %rax,%r9
    3757:	0f 97 45 b7          	seta   -0x49(%rbp)
    375b:	4d 29 d1             	sub    %r10,%r9
    375e:	49 83 c1 0c          	add    $0xc,%r9
    3762:	49 83 f9 18          	cmp    $0x18,%r9
    3766:	4d 8d 48 20          	lea    0x20(%r8),%r9
    376a:	41 0f 97 c2          	seta   %r10b
    376e:	44 22 55 b7          	and    -0x49(%rbp),%r10b
    3772:	4c 39 ce             	cmp    %r9,%rsi
    3775:	41 0f 9d c1          	setge  %r9b
    3779:	48 83 c6 20          	add    $0x20,%rsi
    377d:	49 39 f0             	cmp    %rsi,%r8
    3780:	40 0f 9d c6          	setge  %sil
    3784:	41 09 f1             	or     %esi,%r9d
    3787:	48 89 c6             	mov    %rax,%rsi
    378a:	45 21 d1             	and    %r10d,%r9d
    378d:	4c 8b 55 b8          	mov    -0x48(%rbp),%r10
    3791:	4c 29 d6             	sub    %r10,%rsi
    3794:	48 83 c6 0c          	add    $0xc,%rsi
    3798:	48 83 fe 18          	cmp    $0x18,%rsi
    379c:	40 0f 97 c6          	seta   %sil
    37a0:	41 84 f1             	test   %sil,%r9b
    37a3:	0f 84 97 01 00 00    	je     3940 <_Z4fftriibP9complex_tS0_+0x3b0>
    37a9:	48 89 d6             	mov    %rdx,%rsi
    37ac:	4c 29 d6             	sub    %r10,%rsi
    37af:	48 83 c6 0c          	add    $0xc,%rsi
    37b3:	48 83 fe 18          	cmp    $0x18,%rsi
    37b7:	0f 86 83 01 00 00    	jbe    3940 <_Z4fftriibP9complex_tS0_+0x3b0>
    37bd:	83 7d b0 01          	cmpl   $0x1,-0x50(%rbp)
    37c1:	0f 84 61 02 00 00    	je     3a28 <_Z4fftriibP9complex_tS0_+0x498>
    37c7:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
    37cb:	c4 41 08 14 cd       	vunpcklps %xmm13,%xmm14,%xmm9
    37d0:	c4 41 10 14 c6       	vunpcklps %xmm14,%xmm13,%xmm8
                    x[q + s * (p + m)] = a - b;
    37d5:	4c 8b 4d 98          	mov    -0x68(%rbp),%r9
    37d9:	c4 c1 30 16 e9       	vmovlhps %xmm9,%xmm9,%xmm5
    37de:	c4 c1 38 16 e0       	vmovlhps %xmm8,%xmm8,%xmm4
    37e3:	49 8d 7c 3e f0       	lea    -0x10(%r14,%rdi,1),%rdi
    37e8:	4d 8d 44 36 f0       	lea    -0x10(%r14,%rsi,1),%r8
    37ed:	c4 e3 55 18 ed 01    	vinsertf128 $0x1,%xmm5,%ymm5,%ymm5
    37f3:	c4 e3 5d 18 e4 01    	vinsertf128 $0x1,%xmm4,%ymm4,%ymm4
    37f9:	31 f6                	xor    %esi,%esi
    37fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3800:	c4 c1 7c 10 04 30    	vmovups (%r8,%rsi,1),%ymm0
                    const complex_t a = y[q + s * (2 * p + 0)];
    3806:	c5 fc 10 0c 37       	vmovups (%rdi,%rsi,1),%ymm1
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    380b:	c4 e3 7d 04 d0 a0    	vpermilps $0xa0,%ymm0,%ymm2
    3811:	c4 e3 7d 04 c0 f5    	vpermilps $0xf5,%ymm0,%ymm0
    3817:	c5 fc 59 c4          	vmulps %ymm4,%ymm0,%ymm0
    381b:	c4 e2 6d b6 c5       	vfmaddsub231ps %ymm5,%ymm2,%ymm0
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3820:	c5 fc 58 d9          	vaddps %ymm1,%ymm0,%ymm3
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3824:	c5 f4 5c c8          	vsubps %ymm0,%ymm1,%ymm1
                    x[q + s * (p + 0)] = a + b;
    3828:	c5 fc 11 1c 30       	vmovups %ymm3,(%rax,%rsi,1)
                    x[q + s * (p + m)] = a - b;
    382d:	c5 fc 11 0c 32       	vmovups %ymm1,(%rdx,%rsi,1)
                for (int q = 0; q < s; q += 2)
    3832:	48 83 c6 20          	add    $0x20,%rsi
    3836:	49 39 f1             	cmp    %rsi,%r9
    3839:	75 c5                	jne    3800 <_Z4fftriibP9complex_tS0_+0x270>
    383b:	8b 45 94             	mov    -0x6c(%rbp),%eax
    383e:	85 c0                	test   %eax,%eax
    3840:	74 60                	je     38a2 <_Z4fftriibP9complex_tS0_+0x312>
    3842:	8b 45 90             	mov    -0x70(%rbp),%eax
                    const complex_t a = y[q + s * (2 * p + 0)];
    3845:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    3849:	48 01 c8             	add    %rcx,%rax
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    384c:	c4 41 38 16 c0       	vmovlhps %xmm8,%xmm8,%xmm8
    3851:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
                    const complex_t a = y[q + s * (2 * p + 0)];
    3855:	48 c1 e0 04          	shl    $0x4,%rax
    3859:	c4 c1 30 16 d9       	vmovlhps %xmm9,%xmm9,%xmm3
    385e:	48 8d 34 11          	lea    (%rcx,%rdx,1),%rsi
    3862:	c4 c1 78 10 24 06    	vmovups (%r14,%rax,1),%xmm4
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3868:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
    386c:	48 01 d7             	add    %rdx,%rdi
    386f:	48 01 d0             	add    %rdx,%rax
    3872:	c4 c1 78 10 0c c6    	vmovups (%r14,%rax,8),%xmm1
    3878:	c4 e3 79 04 e9 a0    	vpermilps $0xa0,%xmm1,%xmm5
    387e:	c4 e3 79 04 c9 f5    	vpermilps $0xf5,%xmm1,%xmm1
    3884:	c4 c1 70 59 c8       	vmulps %xmm8,%xmm1,%xmm1
    3889:	c4 e2 51 b6 cb       	vfmaddsub231ps %xmm3,%xmm5,%xmm1
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    388e:	c5 f0 58 dc          	vaddps %xmm4,%xmm1,%xmm3
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3892:	c5 d8 5c e1          	vsubps %xmm1,%xmm4,%xmm4
                    x[q + s * (p + 0)] = a + b;
    3896:	c4 c1 78 11 1c f3    	vmovups %xmm3,(%r11,%rsi,8)
                    x[q + s * (p + m)] = a - b;
    389c:	c4 c1 78 11 24 fb    	vmovups %xmm4,(%r11,%rdi,8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    38a2:	c5 92 59 ce          	vmulss %xmm6,%xmm13,%xmm1
        for (int p = 0; p < m; p++)
    38a6:	01 5d c4             	add    %ebx,-0x3c(%rbp)
    38a9:	49 ff c5             	inc    %r13
    38ac:	49 83 c7 10          	add    $0x10,%r15
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    38b0:	c5 92 59 c7          	vmulss %xmm7,%xmm13,%xmm0
        for (int p = 0; p < m; p++)
    38b4:	41 01 dc             	add    %ebx,%r12d
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    38b7:	c4 c2 41 b9 ce       	vfmadd231ss %xmm14,%xmm7,%xmm1
    38bc:	c4 62 79 9b f6       	vfmsub132ss %xmm6,%xmm0,%xmm14
        for (int p = 0; p < m; p++)
    38c1:	44 39 6d c0          	cmp    %r13d,-0x40(%rbp)
    38c5:	0f 8e 55 01 00 00    	jle    3a20 <_Z4fftriibP9complex_tS0_+0x490>
    38cb:	c5 78 28 e9          	vmovaps %xmm1,%xmm13
            if (s == 1)
    38cf:	83 fb 01             	cmp    $0x1,%ebx
    38d2:	0f 85 18 fe ff ff    	jne    36f0 <_Z4fftriibP9complex_tS0_+0x160>
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    38d8:	c4 41 7a 10 47 0c    	vmovss 0xc(%r15),%xmm8
    38de:	c4 c1 7a 10 4f 08    	vmovss 0x8(%r15),%xmm1
                    const complex_t a = y[q + s * (2 * p + 0)];
    38e4:	c4 c1 7a 10 2f       	vmovss (%r15),%xmm5
    38e9:	c4 c1 7a 10 5f 04    	vmovss 0x4(%r15),%xmm3
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    38ef:	c4 c1 3a 59 e6       	vmulss %xmm14,%xmm8,%xmm4
                    x[q + s * (p + m)] = a - b;
    38f4:	8b 45 c0             	mov    -0x40(%rbp),%eax
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    38f7:	c4 41 12 59 c0       	vmulss %xmm8,%xmm13,%xmm8
                    x[q + s * (p + m)] = a - b;
    38fc:	44 01 e8             	add    %r13d,%eax
    38ff:	89 c0                	mov    %eax,%eax
    3901:	49 8d 04 c3          	lea    (%r11,%rax,8),%rax
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3905:	c4 e2 11 b9 e1       	vfmadd231ss %xmm1,%xmm13,%xmm4
    390a:	c4 c2 39 9b ce       	vfmsub132ss %xmm14,%xmm8,%xmm1
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    390f:	c5 52 58 c1          	vaddss %xmm1,%xmm5,%xmm8
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3913:	c5 d2 5c c9          	vsubss %xmm1,%xmm5,%xmm1
                    x[q + s * (p + 0)] = a + b;
    3917:	c4 01 7a 11 04 eb    	vmovss %xmm8,(%r11,%r13,8)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    391d:	c5 62 58 c4          	vaddss %xmm4,%xmm3,%xmm8
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3921:	c5 e2 5c dc          	vsubss %xmm4,%xmm3,%xmm3
                    x[q + s * (p + 0)] = a + b;
    3925:	c4 01 7a 11 44 eb 04 	vmovss %xmm8,0x4(%r11,%r13,8)
                    x[q + s * (p + m)] = a - b;
    392c:	c5 fa 11 08          	vmovss %xmm1,(%rax)
    3930:	c5 fa 11 58 04       	vmovss %xmm3,0x4(%rax)
                for (int q = 0; q < s; q++)
    3935:	e9 68 ff ff ff       	jmp    38a2 <_Z4fftriibP9complex_tS0_+0x312>
    393a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    3940:	4c 8b 55 88          	mov    -0x78(%rbp),%r10
    3944:	4c 8b 45 80          	mov    -0x80(%rbp),%r8
    3948:	49 8d 74 3e f0       	lea    -0x10(%r14,%rdi,1),%rsi
    394d:	48 8b 7d c8          	mov    -0x38(%rbp),%rdi
    3951:	4c 01 d1             	add    %r10,%rcx
    3954:	48 c1 e1 04          	shl    $0x4,%rcx
    3958:	49 8d 7c 3e f0       	lea    -0x10(%r14,%rdi,1),%rdi
    395d:	4c 01 c1             	add    %r8,%rcx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3960:	c5 fa 10 1f          	vmovss (%rdi),%xmm3
                    const complex_t a = y[q + s * (2 * p + 0)];
    3964:	c5 7a 10 1e          	vmovss (%rsi),%xmm11
                for (int q = 0; q < s; q += 2)
    3968:	48 83 c6 10          	add    $0x10,%rsi
    396c:	48 83 c7 10          	add    $0x10,%rdi
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3970:	c5 fa 10 4f f4       	vmovss -0xc(%rdi),%xmm1
    3975:	c5 7a 10 67 fc       	vmovss -0x4(%rdi),%xmm12
                for (int q = 0; q < s; q += 2)
    397a:	48 83 c0 10          	add    $0x10,%rax
    397e:	48 83 c2 10          	add    $0x10,%rdx
                    const complex_t a = y[q + s * (2 * p + 0)];
    3982:	c5 7a 10 4e f4       	vmovss -0xc(%rsi),%xmm9
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3987:	c5 7a 10 46 f8       	vmovss -0x8(%rsi),%xmm8
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    398c:	c5 0a 59 d1          	vmulss %xmm1,%xmm14,%xmm10
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3990:	c5 fa 10 66 fc       	vmovss -0x4(%rsi),%xmm4
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3995:	c5 92 59 c9          	vmulss %xmm1,%xmm13,%xmm1
    3999:	c4 c1 0a 59 ec       	vmulss %xmm12,%xmm14,%xmm5
    399e:	c4 41 12 59 e4       	vmulss %xmm12,%xmm13,%xmm12
    39a3:	c4 62 11 b9 d3       	vfmadd231ss %xmm3,%xmm13,%xmm10
    39a8:	c4 c2 71 9b de       	vfmsub132ss %xmm14,%xmm1,%xmm3
    39ad:	c5 fa 10 4f f8       	vmovss -0x8(%rdi),%xmm1
    39b2:	c4 e2 11 b9 e9       	vfmadd231ss %xmm1,%xmm13,%xmm5
    39b7:	c4 c2 19 9b ce       	vfmsub132ss %xmm14,%xmm12,%xmm1
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    39bc:	c4 41 62 58 e3       	vaddss %xmm11,%xmm3,%xmm12
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    39c1:	c5 a2 5c db          	vsubss %xmm3,%xmm11,%xmm3
                    x[q + s * (p + 0)] = a + b;
    39c5:	c5 7a 11 60 f0       	vmovss %xmm12,-0x10(%rax)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    39ca:	c4 41 2a 58 e1       	vaddss %xmm9,%xmm10,%xmm12
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    39cf:	c4 41 32 5c ca       	vsubss %xmm10,%xmm9,%xmm9
                    x[q + s * (p + 0)] = a + b;
    39d4:	c5 7a 11 60 f4       	vmovss %xmm12,-0xc(%rax)
                    x[q + s * (p + m)] = a - b;
    39d9:	c5 fa 11 5a f0       	vmovss %xmm3,-0x10(%rdx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    39de:	c4 c1 72 58 d8       	vaddss %xmm8,%xmm1,%xmm3
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    39e3:	c5 ba 5c c9          	vsubss %xmm1,%xmm8,%xmm1
                    x[q + s * (p + m)] = a - b;
    39e7:	c5 7a 11 4a f4       	vmovss %xmm9,-0xc(%rdx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    39ec:	c5 fa 11 58 f8       	vmovss %xmm3,-0x8(%rax)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    39f1:	c5 d2 58 dc          	vaddss %xmm4,%xmm5,%xmm3
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    39f5:	c5 da 5c e5          	vsubss %xmm5,%xmm4,%xmm4
                    x[(q + 1) + s * (p + 0)] = c + d;
    39f9:	c5 fa 11 58 fc       	vmovss %xmm3,-0x4(%rax)
                    x[(q + 1) + s * (p + m)] = c - d;
    39fe:	c5 fa 11 4a f8       	vmovss %xmm1,-0x8(%rdx)
    3a03:	c5 fa 11 62 fc       	vmovss %xmm4,-0x4(%rdx)
                for (int q = 0; q < s; q += 2)
    3a08:	48 39 ce             	cmp    %rcx,%rsi
    3a0b:	0f 85 4f ff ff ff    	jne    3960 <_Z4fftriibP9complex_tS0_+0x3d0>
    3a11:	e9 8c fe ff ff       	jmp    38a2 <_Z4fftriibP9complex_tS0_+0x312>
    3a16:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3a1d:	00 00 00 
    3a20:	c5 f8 77             	vzeroupper
    3a23:	e9 c9 fb ff ff       	jmp    35f1 <_Z4fftriibP9complex_tS0_+0x61>
                    x[q + s * (p + m)] = a - b;
    3a28:	31 c0                	xor    %eax,%eax
    3a2a:	c4 41 08 14 cd       	vunpcklps %xmm13,%xmm14,%xmm9
    3a2f:	c4 41 10 14 c6       	vunpcklps %xmm14,%xmm13,%xmm8
    3a34:	e9 0c fe ff ff       	jmp    3845 <_Z4fftriibP9complex_tS0_+0x2b5>
    3a39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000003a40 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_>:
{
    3a40:	41 55                	push   %r13
    3a42:	4c 8d 6c 24 10       	lea    0x10(%rsp),%r13
    3a47:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    3a4b:	41 ff 75 f8          	push   -0x8(%r13)
    3a4f:	55                   	push   %rbp
    3a50:	48 89 e5             	mov    %rsp,%rbp
    3a53:	41 57                	push   %r15
    3a55:	41 56                	push   %r14
    3a57:	49 89 f6             	mov    %rsi,%r14
    3a5a:	41 55                	push   %r13
    3a5c:	41 54                	push   %r12
    3a5e:	53                   	push   %rbx
    3a5f:	48 83 ec 68          	sub    $0x68,%rsp
    3a63:	4c 8b 3a             	mov    (%rdx),%r15
    3a66:	48 8b 1e             	mov    (%rsi),%rbx
    if (n == 1)
    3a69:	83 ff 01             	cmp    $0x1,%edi
    3a6c:	75 32                	jne    3aa0 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x60>
	  const ptrdiff_t _Num = __last - __first;
    3a6e:	49 8b 56 08          	mov    0x8(%r14),%rdx
    3a72:	48 29 da             	sub    %rbx,%rdx
	  if (__builtin_expect(_Num > 1, true))
    3a75:	48 83 fa 08          	cmp    $0x8,%rdx
    3a79:	0f 8e 41 01 00 00    	jle    3bc0 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x180>
}
    3a7f:	48 83 c4 68          	add    $0x68,%rsp
	    __builtin_memmove(__result, __first, sizeof(_Tp) * _Num);
    3a83:	48 89 de             	mov    %rbx,%rsi
    3a86:	4c 89 ff             	mov    %r15,%rdi
    3a89:	5b                   	pop    %rbx
    3a8a:	41 5c                	pop    %r12
    3a8c:	41 5d                	pop    %r13
    3a8e:	41 5e                	pop    %r14
    3a90:	41 5f                	pop    %r15
    3a92:	5d                   	pop    %rbp
    3a93:	49 8d 65 f0          	lea    -0x10(%r13),%rsp
    3a97:	41 5d                	pop    %r13
    3a99:	e9 72 e7 ff ff       	jmp    2210 <memmove@plt>
    3a9e:	66 90                	xchg   %ax,%ax
    const int m = n / 2;
    3aa0:	41 89 fc             	mov    %edi,%r12d
    3aa3:	49 89 fd             	mov    %rdi,%r13
    3aa6:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    3aab:	41 c1 ec 1f          	shr    $0x1f,%r12d
    3aaf:	41 01 fc             	add    %edi,%r12d
    3ab2:	41 d1 fc             	sar    $1,%r12d
    if (n == 1)
    3ab5:	41 83 fc 01          	cmp    $0x1,%r12d
    3ab9:	0f 85 31 01 00 00    	jne    3bf0 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1b0>
                x[q] = y[q];
    3abf:	48 8b 03             	mov    (%rbx),%rax
    3ac2:	c5 fa 10 15 26 26 00 	vmovss 0x2626(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    3ac9:	00 
    3aca:	c5 fb 10 2d 0e 26 00 	vmovsd 0x260e(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    3ad1:	00 
    3ad2:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
    3ad7:	49 89 07             	mov    %rax,(%r15)
    3ada:	48 8b 43 08          	mov    0x8(%rbx),%rax
    3ade:	49 89 47 08          	mov    %rax,0x8(%r15)
    const float angle = 2 * M_PI / n;
    3ae2:	c4 41 3b 2a c5       	vcvtsi2sd %r13d,%xmm8,%xmm8
    3ae7:	c4 41 53 5e c0       	vdivsd %xmm8,%xmm5,%xmm8
    3aec:	c4 41 3b 5a c0       	vcvtsd2ss %xmm8,%xmm8,%xmm8
  { return __builtin_cosf(__x); }
    3af1:	c5 78 29 c0          	vmovaps %xmm8,%xmm0
    3af5:	c5 7a 11 45 bc       	vmovss %xmm8,-0x44(%rbp)
    3afa:	e8 01 e6 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    3aff:	c5 7a 10 45 bc       	vmovss -0x44(%rbp),%xmm8
    3b04:	c5 fa 11 45 b0       	vmovss %xmm0,-0x50(%rbp)
    3b09:	c5 b8 57 45 c0       	vxorps -0x40(%rbp),%xmm8,%xmm0
  { return __builtin_sinf(__x); }
    3b0e:	e8 0d e6 ff ff       	call   2120 <sinf@plt>
    3b13:	c5 78 28 c0          	vmovaps %xmm0,%xmm8
        for (int p = 0; p < m; p++)
    3b17:	41 83 fd 01          	cmp    $0x1,%r13d
    3b1b:	0f 8e 4d ff ff ff    	jle    3a6e <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x2e>
    3b21:	c5 fa 10 15 e3 24 00 	vmovss 0x24e3(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    3b28:	00 
    3b29:	c5 fa 10 7d b0       	vmovss -0x50(%rbp),%xmm7
    3b2e:	49 63 c4             	movslq %r12d,%rax
    3b31:	4c 89 fa             	mov    %r15,%rdx
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    3b34:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    3b38:	48 8d 0c c3          	lea    (%rbx,%rax,8),%rcx
        for (int p = 0; p < m; p++)
    3b3c:	31 c0                	xor    %eax,%eax
    3b3e:	66 90                	xchg   %ax,%ax
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3b40:	c5 fa 10 72 0c       	vmovss 0xc(%rdx),%xmm6
    3b45:	c5 fa 10 4a 08       	vmovss 0x8(%rdx),%xmm1
        for (int p = 0; p < m; p++)
    3b4a:	48 83 c2 10          	add    $0x10,%rdx
                    const complex_t a = y[q + s * (2 * p + 0)];
    3b4e:	c5 fa 10 6a f0       	vmovss -0x10(%rdx),%xmm5
    3b53:	c5 fa 10 5a f4       	vmovss -0xc(%rdx),%xmm3
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3b58:	c5 ea 59 e6          	vmulss %xmm6,%xmm2,%xmm4
    3b5c:	c5 fa 59 f6          	vmulss %xmm6,%xmm0,%xmm6
    3b60:	c4 e2 79 b9 e1       	vfmadd231ss %xmm1,%xmm0,%xmm4
    3b65:	c4 e2 49 9b ca       	vfmsub132ss %xmm2,%xmm6,%xmm1
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3b6a:	c5 d2 58 f1          	vaddss %xmm1,%xmm5,%xmm6
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3b6e:	c5 d2 5c c9          	vsubss %xmm1,%xmm5,%xmm1
                    x[q + s * (p + 0)] = a + b;
    3b72:	c5 fa 11 34 c3       	vmovss %xmm6,(%rbx,%rax,8)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3b77:	c5 e2 58 f4          	vaddss %xmm4,%xmm3,%xmm6
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3b7b:	c5 e2 5c dc          	vsubss %xmm4,%xmm3,%xmm3
                    x[q + s * (p + 0)] = a + b;
    3b7f:	c5 fa 11 74 c3 04    	vmovss %xmm6,0x4(%rbx,%rax,8)
                    x[q + s * (p + m)] = a - b;
    3b85:	c5 fa 11 0c c1       	vmovss %xmm1,(%rcx,%rax,8)
    3b8a:	c5 f8 28 c8          	vmovaps %xmm0,%xmm1
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3b8e:	c4 c1 72 59 c8       	vmulss %xmm8,%xmm1,%xmm1
                    x[q + s * (p + m)] = a - b;
    3b93:	c5 fa 11 5c c1 04    	vmovss %xmm3,0x4(%rcx,%rax,8)
        for (int p = 0; p < m; p++)
    3b99:	48 ff c0             	inc    %rax
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3b9c:	c5 fa 59 c7          	vmulss %xmm7,%xmm0,%xmm0
    3ba0:	c4 c2 69 b9 c0       	vfmadd231ss %xmm8,%xmm2,%xmm0
    3ba5:	c4 e2 71 9b d7       	vfmsub132ss %xmm7,%xmm1,%xmm2
        for (int p = 0; p < m; p++)
    3baa:	41 39 c4             	cmp    %eax,%r12d
    3bad:	7f 91                	jg     3b40 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x100>
	  const ptrdiff_t _Num = __last - __first;
    3baf:	49 8b 56 08          	mov    0x8(%r14),%rdx
    3bb3:	48 29 da             	sub    %rbx,%rdx
	  if (__builtin_expect(_Num > 1, true))
    3bb6:	48 83 fa 08          	cmp    $0x8,%rdx
    3bba:	0f 8f bf fe ff ff    	jg     3a7f <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x3f>
	  else if (_Num == 1)
    3bc0:	75 14                	jne    3bd6 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x196>
	{ *__to = *__from; }
    3bc2:	c5 fa 10 03          	vmovss (%rbx),%xmm0
    3bc6:	c4 c1 7a 11 07       	vmovss %xmm0,(%r15)
    3bcb:	c5 fa 10 43 04       	vmovss 0x4(%rbx),%xmm0
    3bd0:	c4 c1 7a 11 47 04    	vmovss %xmm0,0x4(%r15)
}
    3bd6:	48 83 c4 68          	add    $0x68,%rsp
    3bda:	5b                   	pop    %rbx
    3bdb:	41 5c                	pop    %r12
    3bdd:	41 5d                	pop    %r13
    3bdf:	41 5e                	pop    %r14
    3be1:	41 5f                	pop    %r15
    3be3:	5d                   	pop    %rbp
    3be4:	49 8d 65 f0          	lea    -0x10(%r13),%rsp
    3be8:	41 5d                	pop    %r13
    3bea:	c3                   	ret
    3beb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    const int m = n / 2;
    3bf0:	85 ff                	test   %edi,%edi
    3bf2:	44 8d 4f 03          	lea    0x3(%rdi),%r9d
    3bf6:	44 0f 49 cf          	cmovns %edi,%r9d
    3bfa:	41 c1 f9 02          	sar    $0x2,%r9d
    if (n == 1)
    3bfe:	41 83 f9 01          	cmp    $0x1,%r9d
    3c02:	0f 85 a8 01 00 00    	jne    3db0 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x370>
    3c08:	c5 fa 10 15 e0 24 00 	vmovss 0x24e0(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    3c0f:	00 
    3c10:	c5 fb 10 2d c8 24 00 	vmovsd 0x24c8(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    3c17:	00 
    3c18:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
    const float angle = 2 * M_PI / n;
    3c1d:	c4 c1 3b 2a cc       	vcvtsi2sd %r12d,%xmm8,%xmm1
    3c22:	c5 fb 11 6d b0       	vmovsd %xmm5,-0x50(%rbp)
    3c27:	44 89 4d a0          	mov    %r9d,-0x60(%rbp)
    3c2b:	c5 d3 5e c9          	vdivsd %xmm1,%xmm5,%xmm1
    3c2f:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
  { return __builtin_cosf(__x); }
    3c33:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    3c37:	c5 fa 11 4d bc       	vmovss %xmm1,-0x44(%rbp)
    3c3c:	e8 bf e4 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    3c41:	c5 fa 10 4d bc       	vmovss -0x44(%rbp),%xmm1
    3c46:	c5 fa 11 45 a8       	vmovss %xmm0,-0x58(%rbp)
    3c4b:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    3c50:	e8 cb e4 ff ff       	call   2120 <sinf@plt>
        for (int p = 0; p < m; p++)
    3c55:	41 83 fd 03          	cmp    $0x3,%r13d
    3c59:	c5 fb 10 6d b0       	vmovsd -0x50(%rbp),%xmm5
    3c5e:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    3c63:	c5 78 28 d0          	vmovaps %xmm0,%xmm10
    3c67:	0f 8e 75 fe ff ff    	jle    3ae2 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xa2>
    3c6d:	c5 fa 10 15 97 23 00 	vmovss 0x2397(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    3c74:	00 
    3c75:	44 8b 4d a0          	mov    -0x60(%rbp),%r9d
    3c79:	c5 7a 10 4d a8       	vmovss -0x58(%rbp),%xmm9
    3c7e:	43 8d 14 09          	lea    (%r9,%r9,1),%edx
    3c82:	48 89 d8             	mov    %rbx,%rax
    3c85:	4c 89 f9             	mov    %r15,%rcx
    3c88:	31 f6                	xor    %esi,%esi
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    3c8a:	c5 f8 28 ca          	vmovaps %xmm2,%xmm1
    3c8e:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
        for (int p = 0; p < m; p++)
    3c92:	c5 79 28 fd          	vmovapd %xmm5,%xmm15
    3c96:	48 63 d2             	movslq %edx,%rdx
    3c99:	49 8d 14 d7          	lea    (%r15,%rdx,8),%rdx
    3c9d:	0f 1f 00             	nopl   (%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3ca0:	c5 fa 10 58 14       	vmovss 0x14(%rax),%xmm3
    3ca5:	c5 fa 10 60 10       	vmovss 0x10(%rax),%xmm4
        for (int p = 0; p < m; p++)
    3caa:	ff c6                	inc    %esi
    3cac:	48 83 c0 20          	add    $0x20,%rax
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3cb0:	c5 7a 10 70 fc       	vmovss -0x4(%rax),%xmm14
                    const complex_t a = y[q + s * (2 * p + 0)];
    3cb5:	c5 7a 10 68 e0       	vmovss -0x20(%rax),%xmm13
        for (int p = 0; p < m; p++)
    3cba:	48 83 c1 10          	add    $0x10,%rcx
    3cbe:	48 83 c2 10          	add    $0x10,%rdx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3cc2:	c5 62 59 e1          	vmulss %xmm1,%xmm3,%xmm12
                    const complex_t a = y[q + s * (2 * p + 0)];
    3cc6:	c5 7a 10 58 e4       	vmovss -0x1c(%rax),%xmm11
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3ccb:	c5 fa 10 78 e8       	vmovss -0x18(%rax),%xmm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3cd0:	c5 fa 59 db          	vmulss %xmm3,%xmm0,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3cd4:	c5 fa 10 68 ec       	vmovss -0x14(%rax),%xmm5
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3cd9:	c5 8a 59 f1          	vmulss %xmm1,%xmm14,%xmm6
    3cdd:	c4 41 7a 59 f6       	vmulss %xmm14,%xmm0,%xmm14
    3ce2:	c4 62 79 b9 e4       	vfmadd231ss %xmm4,%xmm0,%xmm12
    3ce7:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    3cec:	c5 fa 10 58 f8       	vmovss -0x8(%rax),%xmm3
    3cf1:	c4 e2 79 b9 f3       	vfmadd231ss %xmm3,%xmm0,%xmm6
    3cf6:	c4 e2 09 9b d9       	vfmsub132ss %xmm1,%xmm14,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3cfb:	c4 41 5a 58 f5       	vaddss %xmm13,%xmm4,%xmm14
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3d00:	c5 92 5c e4          	vsubss %xmm4,%xmm13,%xmm4
                    x[q + s * (p + 0)] = a + b;
    3d04:	c5 7a 11 71 f0       	vmovss %xmm14,-0x10(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3d09:	c4 41 22 58 f4       	vaddss %xmm12,%xmm11,%xmm14
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3d0e:	c4 41 22 5c dc       	vsubss %xmm12,%xmm11,%xmm11
                    x[q + s * (p + 0)] = a + b;
    3d13:	c5 7a 11 71 f4       	vmovss %xmm14,-0xc(%rcx)
                    x[q + s * (p + m)] = a - b;
    3d18:	c5 fa 11 62 f0       	vmovss %xmm4,-0x10(%rdx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3d1d:	c5 c2 58 e3          	vaddss %xmm3,%xmm7,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3d21:	c5 c2 5c db          	vsubss %xmm3,%xmm7,%xmm3
                    x[q + s * (p + m)] = a - b;
    3d25:	c5 7a 11 5a f4       	vmovss %xmm11,-0xc(%rdx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    3d2a:	c5 fa 11 61 f8       	vmovss %xmm4,-0x8(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3d2f:	c5 d2 58 e6          	vaddss %xmm6,%xmm5,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3d33:	c5 d2 5c ee          	vsubss %xmm6,%xmm5,%xmm5
                    x[(q + 1) + s * (p + 0)] = c + d;
    3d37:	c5 fa 11 61 fc       	vmovss %xmm4,-0x4(%rcx)
                    x[(q + 1) + s * (p + m)] = c - d;
    3d3c:	c5 fa 11 5a f8       	vmovss %xmm3,-0x8(%rdx)
    3d41:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3d45:	c4 c1 62 59 da       	vmulss %xmm10,%xmm3,%xmm3
                    x[(q + 1) + s * (p + m)] = c - d;
    3d4a:	c5 fa 11 6a fc       	vmovss %xmm5,-0x4(%rdx)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3d4f:	c4 c1 7a 59 c1       	vmulss %xmm9,%xmm0,%xmm0
    3d54:	c4 e2 29 b9 c1       	vfmadd231ss %xmm1,%xmm10,%xmm0
    3d59:	c4 c2 61 9b c9       	vfmsub132ss %xmm9,%xmm3,%xmm1
        for (int p = 0; p < m; p++)
    3d5e:	41 39 f1             	cmp    %esi,%r9d
    3d61:	0f 8f 39 ff ff ff    	jg     3ca0 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x260>
    const float angle = 2 * M_PI / n;
    3d67:	c4 41 3b 2a c5       	vcvtsi2sd %r13d,%xmm8,%xmm8
    3d6c:	c5 fa 11 55 a8       	vmovss %xmm2,-0x58(%rbp)
    3d71:	c4 41 03 5e c0       	vdivsd %xmm8,%xmm15,%xmm8
    3d76:	c4 41 3b 5a c0       	vcvtsd2ss %xmm8,%xmm8,%xmm8
  { return __builtin_cosf(__x); }
    3d7b:	c5 78 29 c0          	vmovaps %xmm8,%xmm0
    3d7f:	c5 7a 11 45 b0       	vmovss %xmm8,-0x50(%rbp)
    3d84:	e8 77 e3 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    3d89:	c5 7a 10 45 b0       	vmovss -0x50(%rbp),%xmm8
    3d8e:	c5 fa 11 45 bc       	vmovss %xmm0,-0x44(%rbp)
    3d93:	c5 b8 57 45 c0       	vxorps -0x40(%rbp),%xmm8,%xmm0
  { return __builtin_sinf(__x); }
    3d98:	e8 83 e3 ff ff       	call   2120 <sinf@plt>
    3d9d:	c5 fa 10 7d bc       	vmovss -0x44(%rbp),%xmm7
    3da2:	c5 fa 10 55 a8       	vmovss -0x58(%rbp),%xmm2
    3da7:	c5 78 28 c0          	vmovaps %xmm0,%xmm8
        for (int p = 0; p < m; p++)
    3dab:	e9 7e fd ff ff       	jmp    3b2e <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xee>
    const int m = n / 2;
    3db0:	85 ff                	test   %edi,%edi
    3db2:	8d 47 07             	lea    0x7(%rdi),%eax
    3db5:	0f 49 c7             	cmovns %edi,%eax
    3db8:	c1 f8 03             	sar    $0x3,%eax
    if (n == 1)
    3dbb:	83 f8 01             	cmp    $0x1,%eax
    3dbe:	0f 85 9c 02 00 00    	jne    4060 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x620>
                x[q] = y[q];
    3dc4:	48 8b 13             	mov    (%rbx),%rdx
    3dc7:	c5 fa 10 15 21 23 00 	vmovss 0x2321(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    3dce:	00 
    3dcf:	c5 fb 10 2d 09 23 00 	vmovsd 0x2309(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    3dd6:	00 
    3dd7:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
    3ddc:	49 89 17             	mov    %rdx,(%r15)
    3ddf:	48 8b 53 08          	mov    0x8(%rbx),%rdx
    3de3:	49 89 57 08          	mov    %rdx,0x8(%r15)
    3de7:	48 8b 53 10          	mov    0x10(%rbx),%rdx
    3deb:	49 89 57 10          	mov    %rdx,0x10(%r15)
    3def:	48 8b 53 18          	mov    0x18(%rbx),%rdx
    3df3:	49 89 57 18          	mov    %rdx,0x18(%r15)
    3df7:	48 8b 53 20          	mov    0x20(%rbx),%rdx
    3dfb:	49 89 57 20          	mov    %rdx,0x20(%r15)
    3dff:	48 8b 53 28          	mov    0x28(%rbx),%rdx
    3e03:	49 89 57 28          	mov    %rdx,0x28(%r15)
    3e07:	48 8b 53 30          	mov    0x30(%rbx),%rdx
    3e0b:	49 89 57 30          	mov    %rdx,0x30(%r15)
    3e0f:	48 8b 53 38          	mov    0x38(%rbx),%rdx
    3e13:	49 89 57 38          	mov    %rdx,0x38(%r15)
    const float angle = 2 * M_PI / n;
    3e17:	c4 c1 3b 2a c9       	vcvtsi2sd %r9d,%xmm8,%xmm1
    3e1c:	44 89 4d a8          	mov    %r9d,-0x58(%rbp)
    3e20:	c5 fb 11 6d b0       	vmovsd %xmm5,-0x50(%rbp)
    3e25:	89 45 98             	mov    %eax,-0x68(%rbp)
    3e28:	c5 d3 5e c9          	vdivsd %xmm1,%xmm5,%xmm1
    3e2c:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
  { return __builtin_cosf(__x); }
    3e30:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    3e34:	c5 fa 11 4d bc       	vmovss %xmm1,-0x44(%rbp)
    3e39:	e8 c2 e2 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    3e3e:	c5 fa 10 4d bc       	vmovss -0x44(%rbp),%xmm1
    3e43:	c5 fa 11 45 a0       	vmovss %xmm0,-0x60(%rbp)
    3e48:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    3e4d:	e8 ce e2 ff ff       	call   2120 <sinf@plt>
        for (int p = 0; p < m; p++)
    3e52:	41 83 fd 07          	cmp    $0x7,%r13d
    3e56:	c5 fb 10 6d b0       	vmovsd -0x50(%rbp),%xmm5
    3e5b:	44 8b 4d a8          	mov    -0x58(%rbp),%r9d
    3e5f:	c5 78 28 d0          	vmovaps %xmm0,%xmm10
    3e63:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    3e68:	0f 8e af fd ff ff    	jle    3c1d <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1dd>
    3e6e:	c5 fa 10 15 96 21 00 	vmovss 0x2196(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    3e75:	00 
    3e76:	8b 45 98             	mov    -0x68(%rbp),%eax
    3e79:	c5 7a 10 4d a0       	vmovss -0x60(%rbp),%xmm9
    3e7e:	8d 0c 85 00 00 00 00 	lea    0x0(,%rax,4),%ecx
    3e85:	4c 89 fa             	mov    %r15,%rdx
    3e88:	48 89 de             	mov    %rbx,%rsi
    3e8b:	31 ff                	xor    %edi,%edi
    3e8d:	48 63 c9             	movslq %ecx,%rcx
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    3e90:	c5 f8 28 ca          	vmovaps %xmm2,%xmm1
    3e94:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    3e98:	48 8d 0c cb          	lea    (%rbx,%rcx,8),%rcx
    3e9c:	0f 1f 40 00          	nopl   0x0(%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3ea0:	c5 fa 10 5a 24       	vmovss 0x24(%rdx),%xmm3
    3ea5:	c5 fa 10 62 20       	vmovss 0x20(%rdx),%xmm4
        for (int p = 0; p < m; p++)
    3eaa:	ff c7                	inc    %edi
    3eac:	48 83 c2 40          	add    $0x40,%rdx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3eb0:	c5 7a 10 7a ec       	vmovss -0x14(%rdx),%xmm15
                    const complex_t a = y[q + s * (2 * p + 0)];
    3eb5:	c5 7a 10 72 c0       	vmovss -0x40(%rdx),%xmm14
        for (int p = 0; p < m; p++)
    3eba:	48 83 c6 20          	add    $0x20,%rsi
    3ebe:	48 83 c1 20          	add    $0x20,%rcx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3ec2:	c5 72 59 eb          	vmulss %xmm3,%xmm1,%xmm13
                    const complex_t a = y[q + s * (2 * p + 0)];
    3ec6:	c5 7a 10 62 c4       	vmovss -0x3c(%rdx),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3ecb:	c5 7a 10 5a c8       	vmovss -0x38(%rdx),%xmm11
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3ed0:	c5 e2 59 d8          	vmulss %xmm0,%xmm3,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3ed4:	c5 fa 10 72 cc       	vmovss -0x34(%rdx),%xmm6
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3ed9:	c4 c1 72 59 ff       	vmulss %xmm15,%xmm1,%xmm7
    3ede:	c5 02 59 f8          	vmulss %xmm0,%xmm15,%xmm15
    3ee2:	c4 62 59 b9 e8       	vfmadd231ss %xmm0,%xmm4,%xmm13
    3ee7:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    3eec:	c5 fa 10 5a e8       	vmovss -0x18(%rdx),%xmm3
    3ef1:	c4 e2 61 b9 f8       	vfmadd231ss %xmm0,%xmm3,%xmm7
    3ef6:	c4 e2 01 9b d9       	vfmsub132ss %xmm1,%xmm15,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3efb:	c5 0a 58 fc          	vaddss %xmm4,%xmm14,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3eff:	c5 8a 5c e4          	vsubss %xmm4,%xmm14,%xmm4
                    x[q + s * (p + 0)] = a + b;
    3f03:	c5 7a 11 7e e0       	vmovss %xmm15,-0x20(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3f08:	c4 41 1a 58 fd       	vaddss %xmm13,%xmm12,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3f0d:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    3f12:	c5 7a 11 7e e4       	vmovss %xmm15,-0x1c(%rsi)
                    x[q + s * (p + m)] = a - b;
    3f17:	c5 fa 11 61 e0       	vmovss %xmm4,-0x20(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3f1c:	c4 c1 62 58 e3       	vaddss %xmm11,%xmm3,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3f21:	c5 a2 5c db          	vsubss %xmm3,%xmm11,%xmm3
                    x[q + s * (p + m)] = a - b;
    3f25:	c5 7a 11 61 e4       	vmovss %xmm12,-0x1c(%rcx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    3f2a:	c5 fa 11 66 e8       	vmovss %xmm4,-0x18(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3f2f:	c5 c2 58 e6          	vaddss %xmm6,%xmm7,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3f33:	c5 ca 5c f7          	vsubss %xmm7,%xmm6,%xmm6
                    x[(q + 1) + s * (p + 0)] = c + d;
    3f37:	c5 fa 11 66 ec       	vmovss %xmm4,-0x14(%rsi)
                    x[(q + 1) + s * (p + m)] = c - d;
    3f3c:	c5 fa 11 59 e8       	vmovss %xmm3,-0x18(%rcx)
    3f41:	c5 fa 11 71 ec       	vmovss %xmm6,-0x14(%rcx)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3f46:	c5 fa 10 5a f4       	vmovss -0xc(%rdx),%xmm3
    3f4b:	c5 fa 10 62 f0       	vmovss -0x10(%rdx),%xmm4
    3f50:	c5 7a 10 7a fc       	vmovss -0x4(%rdx),%xmm15
                    const complex_t a = y[q + s * (2 * p + 0)];
    3f55:	c5 7a 10 72 d0       	vmovss -0x30(%rdx),%xmm14
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3f5a:	c5 72 59 eb          	vmulss %xmm3,%xmm1,%xmm13
                    const complex_t a = y[q + s * (2 * p + 0)];
    3f5e:	c5 7a 10 62 d4       	vmovss -0x2c(%rdx),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3f63:	c5 7a 10 5a d8       	vmovss -0x28(%rdx),%xmm11
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3f68:	c5 e2 59 d8          	vmulss %xmm0,%xmm3,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    3f6c:	c5 fa 10 72 dc       	vmovss -0x24(%rdx),%xmm6
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3f71:	c4 c1 72 59 ff       	vmulss %xmm15,%xmm1,%xmm7
    3f76:	c5 02 59 f8          	vmulss %xmm0,%xmm15,%xmm15
    3f7a:	c4 62 59 b9 e8       	vfmadd231ss %xmm0,%xmm4,%xmm13
    3f7f:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    3f84:	c5 fa 10 5a f8       	vmovss -0x8(%rdx),%xmm3
    3f89:	c4 e2 61 b9 f8       	vfmadd231ss %xmm0,%xmm3,%xmm7
    3f8e:	c4 e2 01 9b d9       	vfmsub132ss %xmm1,%xmm15,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3f93:	c5 0a 58 fc          	vaddss %xmm4,%xmm14,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3f97:	c5 8a 5c e4          	vsubss %xmm4,%xmm14,%xmm4
                    x[q + s * (p + 0)] = a + b;
    3f9b:	c5 7a 11 7e f0       	vmovss %xmm15,-0x10(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3fa0:	c4 41 1a 58 fd       	vaddss %xmm13,%xmm12,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3fa5:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    3faa:	c5 7a 11 7e f4       	vmovss %xmm15,-0xc(%rsi)
                    x[q + s * (p + m)] = a - b;
    3faf:	c5 fa 11 61 f0       	vmovss %xmm4,-0x10(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3fb4:	c4 c1 62 58 e3       	vaddss %xmm11,%xmm3,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3fb9:	c5 a2 5c db          	vsubss %xmm3,%xmm11,%xmm3
                    x[q + s * (p + m)] = a - b;
    3fbd:	c5 7a 11 61 f4       	vmovss %xmm12,-0xc(%rcx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    3fc2:	c5 fa 11 66 f8       	vmovss %xmm4,-0x8(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    3fc7:	c5 c2 58 e6          	vaddss %xmm6,%xmm7,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    3fcb:	c5 ca 5c f7          	vsubss %xmm7,%xmm6,%xmm6
                    x[(q + 1) + s * (p + 0)] = c + d;
    3fcf:	c5 fa 11 66 fc       	vmovss %xmm4,-0x4(%rsi)
                    x[(q + 1) + s * (p + m)] = c - d;
    3fd4:	c5 fa 11 59 f8       	vmovss %xmm3,-0x8(%rcx)
    3fd9:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3fdd:	c4 c1 62 59 da       	vmulss %xmm10,%xmm3,%xmm3
                    x[(q + 1) + s * (p + m)] = c - d;
    3fe2:	c5 fa 11 71 fc       	vmovss %xmm6,-0x4(%rcx)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    3fe7:	c4 c1 7a 59 c1       	vmulss %xmm9,%xmm0,%xmm0
    3fec:	c4 c2 71 b9 c2       	vfmadd231ss %xmm10,%xmm1,%xmm0
    3ff1:	c4 c2 61 9b c9       	vfmsub132ss %xmm9,%xmm3,%xmm1
        for (int p = 0; p < m; p++)
    3ff6:	39 f8                	cmp    %edi,%eax
    3ff8:	0f 8f a2 fe ff ff    	jg     3ea0 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x460>
    const float angle = 2 * M_PI / n;
    3ffe:	c4 c1 3b 2a cc       	vcvtsi2sd %r12d,%xmm8,%xmm1
    4003:	44 89 4d a0          	mov    %r9d,-0x60(%rbp)
    4007:	c5 fa 11 55 98       	vmovss %xmm2,-0x68(%rbp)
    400c:	c5 fb 11 6d a8       	vmovsd %xmm5,-0x58(%rbp)
    4011:	c5 d3 5e c9          	vdivsd %xmm1,%xmm5,%xmm1
    4015:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
  { return __builtin_cosf(__x); }
    4019:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    401d:	c5 fa 11 4d b0       	vmovss %xmm1,-0x50(%rbp)
    4022:	e8 d9 e0 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    4027:	c5 fa 10 4d b0       	vmovss -0x50(%rbp),%xmm1
    402c:	c5 fa 11 45 bc       	vmovss %xmm0,-0x44(%rbp)
    4031:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    4036:	e8 e5 e0 ff ff       	call   2120 <sinf@plt>
    403b:	c5 7a 10 4d bc       	vmovss -0x44(%rbp),%xmm9
    4040:	c5 fb 10 6d a8       	vmovsd -0x58(%rbp),%xmm5
    4045:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    404a:	44 8b 4d a0          	mov    -0x60(%rbp),%r9d
    404e:	c5 fa 10 55 98       	vmovss -0x68(%rbp),%xmm2
    4053:	c5 78 28 d0          	vmovaps %xmm0,%xmm10
        for (int p = 0; p < m; p++)
    4057:	e9 22 fc ff ff       	jmp    3c7e <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x23e>
    405c:	0f 1f 40 00          	nopl   0x0(%rax)
    const int m = n / 2;
    4060:	85 ff                	test   %edi,%edi
    4062:	8d 57 0f             	lea    0xf(%rdi),%edx
    4065:	0f 49 d7             	cmovns %edi,%edx
    4068:	c1 fa 04             	sar    $0x4,%edx
    406b:	41 89 d3             	mov    %edx,%r11d
    if (n == 1)
    406e:	83 fa 01             	cmp    $0x1,%edx
    4071:	0f 85 9f 03 00 00    	jne    4416 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x9d6>
    4077:	c5 fa 10 15 71 20 00 	vmovss 0x2071(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    407e:	00 
    407f:	c5 fb 10 2d 59 20 00 	vmovsd 0x2059(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    4086:	00 
    4087:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
    const float angle = 2 * M_PI / n;
    408c:	c5 bb 2a c0          	vcvtsi2sd %eax,%xmm8,%xmm0
    4090:	44 89 4d a0          	mov    %r9d,-0x60(%rbp)
    4094:	89 45 a8             	mov    %eax,-0x58(%rbp)
    4097:	c5 fb 11 6d b0       	vmovsd %xmm5,-0x50(%rbp)
    409c:	44 89 5d 90          	mov    %r11d,-0x70(%rbp)
    40a0:	c5 d3 5e c0          	vdivsd %xmm0,%xmm5,%xmm0
    40a4:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  { return __builtin_cosf(__x); }
    40a8:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    40ac:	c5 fa 11 4d bc       	vmovss %xmm1,-0x44(%rbp)
    40b1:	e8 4a e0 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    40b6:	c5 fa 10 4d bc       	vmovss -0x44(%rbp),%xmm1
    40bb:	c5 fa 11 45 98       	vmovss %xmm0,-0x68(%rbp)
    40c0:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    40c5:	e8 56 e0 ff ff       	call   2120 <sinf@plt>
        for (int p = 0; p < m; p++)
    40ca:	41 83 fd 0f          	cmp    $0xf,%r13d
    40ce:	8b 45 a8             	mov    -0x58(%rbp),%eax
    40d1:	44 8b 4d a0          	mov    -0x60(%rbp),%r9d
    40d5:	c5 fb 10 6d b0       	vmovsd -0x50(%rbp),%xmm5
    40da:	c5 f8 28 f8          	vmovaps %xmm0,%xmm7
    40de:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    40e3:	0f 8e 2e fd ff ff    	jle    3e17 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x3d7>
    40e9:	c5 fa 10 15 1b 1f 00 	vmovss 0x1f1b(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    40f0:	00 
    40f1:	44 8b 5d 90          	mov    -0x70(%rbp),%r11d
    40f5:	c5 fa 10 75 98       	vmovss -0x68(%rbp),%xmm6
    40fa:	42 8d 0c dd 00 00 00 	lea    0x0(,%r11,8),%ecx
    4101:	00 
    4102:	48 89 da             	mov    %rbx,%rdx
    4105:	4c 89 fe             	mov    %r15,%rsi
    4108:	31 ff                	xor    %edi,%edi
    410a:	48 63 c9             	movslq %ecx,%rcx
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    410d:	c5 f8 28 ca          	vmovaps %xmm2,%xmm1
    4111:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    4115:	49 8d 0c cf          	lea    (%r15,%rcx,8),%rcx
    4119:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4120:	c5 fa 10 5a 44       	vmovss 0x44(%rdx),%xmm3
    4125:	c5 fa 10 62 40       	vmovss 0x40(%rdx),%xmm4
        for (int p = 0; p < m; p++)
    412a:	ff c7                	inc    %edi
    412c:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4130:	c5 7a 10 7a cc       	vmovss -0x34(%rdx),%xmm15
                    const complex_t a = y[q + s * (2 * p + 0)];
    4135:	c5 7a 10 72 80       	vmovss -0x80(%rdx),%xmm14
        for (int p = 0; p < m; p++)
    413a:	48 83 c6 40          	add    $0x40,%rsi
    413e:	48 83 c1 40          	add    $0x40,%rcx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4142:	c5 62 59 e9          	vmulss %xmm1,%xmm3,%xmm13
                    const complex_t a = y[q + s * (2 * p + 0)];
    4146:	c5 7a 10 62 84       	vmovss -0x7c(%rdx),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    414b:	c5 7a 10 5a 88       	vmovss -0x78(%rdx),%xmm11
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4150:	c5 e2 59 d8          	vmulss %xmm0,%xmm3,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    4154:	c5 7a 10 4a 8c       	vmovss -0x74(%rdx),%xmm9
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4159:	c5 02 59 d1          	vmulss %xmm1,%xmm15,%xmm10
    415d:	c5 02 59 f8          	vmulss %xmm0,%xmm15,%xmm15
    4161:	c4 62 59 b9 e8       	vfmadd231ss %xmm0,%xmm4,%xmm13
    4166:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    416b:	c5 fa 10 5a c8       	vmovss -0x38(%rdx),%xmm3
    4170:	c4 62 61 b9 d0       	vfmadd231ss %xmm0,%xmm3,%xmm10
    4175:	c4 e2 01 9b d9       	vfmsub132ss %xmm1,%xmm15,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    417a:	c5 0a 58 fc          	vaddss %xmm4,%xmm14,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    417e:	c5 8a 5c e4          	vsubss %xmm4,%xmm14,%xmm4
                    x[q + s * (p + 0)] = a + b;
    4182:	c5 7a 11 7e c0       	vmovss %xmm15,-0x40(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4187:	c4 41 1a 58 fd       	vaddss %xmm13,%xmm12,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    418c:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    4191:	c5 7a 11 7e c4       	vmovss %xmm15,-0x3c(%rsi)
                    x[q + s * (p + m)] = a - b;
    4196:	c5 fa 11 61 c0       	vmovss %xmm4,-0x40(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    419b:	c4 c1 62 58 e3       	vaddss %xmm11,%xmm3,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    41a0:	c5 a2 5c db          	vsubss %xmm3,%xmm11,%xmm3
                    x[q + s * (p + m)] = a - b;
    41a4:	c5 7a 11 61 c4       	vmovss %xmm12,-0x3c(%rcx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    41a9:	c5 fa 11 66 c8       	vmovss %xmm4,-0x38(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    41ae:	c4 c1 32 58 e2       	vaddss %xmm10,%xmm9,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    41b3:	c4 41 32 5c ca       	vsubss %xmm10,%xmm9,%xmm9
                    x[(q + 1) + s * (p + 0)] = c + d;
    41b8:	c5 fa 11 66 cc       	vmovss %xmm4,-0x34(%rsi)
                    x[(q + 1) + s * (p + m)] = c - d;
    41bd:	c5 fa 11 59 c8       	vmovss %xmm3,-0x38(%rcx)
    41c2:	c5 7a 11 49 cc       	vmovss %xmm9,-0x34(%rcx)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    41c7:	c5 fa 10 5a d4       	vmovss -0x2c(%rdx),%xmm3
    41cc:	c5 fa 10 62 d0       	vmovss -0x30(%rdx),%xmm4
    41d1:	c5 7a 10 7a dc       	vmovss -0x24(%rdx),%xmm15
                    const complex_t a = y[q + s * (2 * p + 0)];
    41d6:	c5 7a 10 72 90       	vmovss -0x70(%rdx),%xmm14
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    41db:	c5 62 59 e9          	vmulss %xmm1,%xmm3,%xmm13
                    const complex_t a = y[q + s * (2 * p + 0)];
    41df:	c5 7a 10 62 94       	vmovss -0x6c(%rdx),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    41e4:	c5 7a 10 5a 98       	vmovss -0x68(%rdx),%xmm11
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    41e9:	c5 e2 59 d8          	vmulss %xmm0,%xmm3,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    41ed:	c5 7a 10 4a 9c       	vmovss -0x64(%rdx),%xmm9
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    41f2:	c5 02 59 d1          	vmulss %xmm1,%xmm15,%xmm10
    41f6:	c5 02 59 f8          	vmulss %xmm0,%xmm15,%xmm15
    41fa:	c4 62 59 b9 e8       	vfmadd231ss %xmm0,%xmm4,%xmm13
    41ff:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    4204:	c5 fa 10 5a d8       	vmovss -0x28(%rdx),%xmm3
    4209:	c4 62 61 b9 d0       	vfmadd231ss %xmm0,%xmm3,%xmm10
    420e:	c4 e2 01 9b d9       	vfmsub132ss %xmm1,%xmm15,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4213:	c5 0a 58 fc          	vaddss %xmm4,%xmm14,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4217:	c5 8a 5c e4          	vsubss %xmm4,%xmm14,%xmm4
                    x[q + s * (p + 0)] = a + b;
    421b:	c5 7a 11 7e d0       	vmovss %xmm15,-0x30(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4220:	c4 41 1a 58 fd       	vaddss %xmm13,%xmm12,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4225:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    422a:	c5 7a 11 7e d4       	vmovss %xmm15,-0x2c(%rsi)
                    x[q + s * (p + m)] = a - b;
    422f:	c5 fa 11 61 d0       	vmovss %xmm4,-0x30(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4234:	c5 a2 58 e3          	vaddss %xmm3,%xmm11,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4238:	c5 a2 5c db          	vsubss %xmm3,%xmm11,%xmm3
                    x[q + s * (p + m)] = a - b;
    423c:	c5 7a 11 61 d4       	vmovss %xmm12,-0x2c(%rcx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    4241:	c5 fa 11 66 d8       	vmovss %xmm4,-0x28(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4246:	c4 c1 32 58 e2       	vaddss %xmm10,%xmm9,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    424b:	c4 41 32 5c ca       	vsubss %xmm10,%xmm9,%xmm9
                    x[(q + 1) + s * (p + 0)] = c + d;
    4250:	c5 fa 11 66 dc       	vmovss %xmm4,-0x24(%rsi)
                    x[(q + 1) + s * (p + m)] = c - d;
    4255:	c5 fa 11 59 d8       	vmovss %xmm3,-0x28(%rcx)
    425a:	c5 7a 11 49 dc       	vmovss %xmm9,-0x24(%rcx)
                    const complex_t a = y[q + s * (2 * p + 0)];
    425f:	c5 7a 10 72 a0       	vmovss -0x60(%rdx),%xmm14
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4264:	c5 fa 10 5a e4       	vmovss -0x1c(%rdx),%xmm3
    4269:	c5 fa 10 62 e0       	vmovss -0x20(%rdx),%xmm4
    426e:	c5 7a 10 7a ec       	vmovss -0x14(%rdx),%xmm15
    4273:	c5 72 59 eb          	vmulss %xmm3,%xmm1,%xmm13
                    const complex_t a = y[q + s * (2 * p + 0)];
    4277:	c5 7a 10 62 a4       	vmovss -0x5c(%rdx),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    427c:	c5 7a 10 5a a8       	vmovss -0x58(%rdx),%xmm11
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4281:	c5 fa 59 db          	vmulss %xmm3,%xmm0,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    4285:	c5 7a 10 4a ac       	vmovss -0x54(%rdx),%xmm9
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    428a:	c4 41 72 59 d7       	vmulss %xmm15,%xmm1,%xmm10
    428f:	c4 41 7a 59 ff       	vmulss %xmm15,%xmm0,%xmm15
    4294:	c4 62 79 b9 ec       	vfmadd231ss %xmm4,%xmm0,%xmm13
    4299:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    429e:	c5 fa 10 5a e8       	vmovss -0x18(%rdx),%xmm3
    42a3:	c4 62 79 b9 d3       	vfmadd231ss %xmm3,%xmm0,%xmm10
    42a8:	c4 e2 01 9b d9       	vfmsub132ss %xmm1,%xmm15,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    42ad:	c5 0a 58 fc          	vaddss %xmm4,%xmm14,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    42b1:	c5 8a 5c e4          	vsubss %xmm4,%xmm14,%xmm4
                    x[q + s * (p + 0)] = a + b;
    42b5:	c5 7a 11 7e e0       	vmovss %xmm15,-0x20(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    42ba:	c4 41 1a 58 fd       	vaddss %xmm13,%xmm12,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    42bf:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    42c4:	c5 7a 11 7e e4       	vmovss %xmm15,-0x1c(%rsi)
                    x[q + s * (p + m)] = a - b;
    42c9:	c5 fa 11 61 e0       	vmovss %xmm4,-0x20(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    42ce:	c5 a2 58 e3          	vaddss %xmm3,%xmm11,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    42d2:	c5 a2 5c db          	vsubss %xmm3,%xmm11,%xmm3
                    x[q + s * (p + m)] = a - b;
    42d6:	c5 7a 11 61 e4       	vmovss %xmm12,-0x1c(%rcx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    42db:	c5 fa 11 66 e8       	vmovss %xmm4,-0x18(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    42e0:	c4 c1 32 58 e2       	vaddss %xmm10,%xmm9,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    42e5:	c4 41 32 5c ca       	vsubss %xmm10,%xmm9,%xmm9
                    x[(q + 1) + s * (p + 0)] = c + d;
    42ea:	c5 fa 11 66 ec       	vmovss %xmm4,-0x14(%rsi)
                    x[(q + 1) + s * (p + m)] = c - d;
    42ef:	c5 fa 11 59 e8       	vmovss %xmm3,-0x18(%rcx)
    42f4:	c5 7a 11 49 ec       	vmovss %xmm9,-0x14(%rcx)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    42f9:	c5 fa 10 5a f4       	vmovss -0xc(%rdx),%xmm3
    42fe:	c5 fa 10 62 f0       	vmovss -0x10(%rdx),%xmm4
    4303:	c5 7a 10 7a fc       	vmovss -0x4(%rdx),%xmm15
                    const complex_t a = y[q + s * (2 * p + 0)];
    4308:	c5 7a 10 72 b0       	vmovss -0x50(%rdx),%xmm14
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    430d:	c5 72 59 eb          	vmulss %xmm3,%xmm1,%xmm13
                    const complex_t a = y[q + s * (2 * p + 0)];
    4311:	c5 7a 10 62 b4       	vmovss -0x4c(%rdx),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    4316:	c5 7a 10 5a b8       	vmovss -0x48(%rdx),%xmm11
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    431b:	c5 fa 59 db          	vmulss %xmm3,%xmm0,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    431f:	c5 7a 10 4a bc       	vmovss -0x44(%rdx),%xmm9
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4324:	c4 41 72 59 d7       	vmulss %xmm15,%xmm1,%xmm10
    4329:	c4 41 7a 59 ff       	vmulss %xmm15,%xmm0,%xmm15
    432e:	c4 62 79 b9 ec       	vfmadd231ss %xmm4,%xmm0,%xmm13
    4333:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    4338:	c5 fa 10 5a f8       	vmovss -0x8(%rdx),%xmm3
    433d:	c4 62 79 b9 d3       	vfmadd231ss %xmm3,%xmm0,%xmm10
    4342:	c4 e2 01 9b d9       	vfmsub132ss %xmm1,%xmm15,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4347:	c5 0a 58 fc          	vaddss %xmm4,%xmm14,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    434b:	c5 8a 5c e4          	vsubss %xmm4,%xmm14,%xmm4
                    x[q + s * (p + 0)] = a + b;
    434f:	c5 7a 11 7e f0       	vmovss %xmm15,-0x10(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4354:	c4 41 1a 58 fd       	vaddss %xmm13,%xmm12,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4359:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    435e:	c5 7a 11 7e f4       	vmovss %xmm15,-0xc(%rsi)
                    x[q + s * (p + m)] = a - b;
    4363:	c5 fa 11 61 f0       	vmovss %xmm4,-0x10(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4368:	c5 a2 58 e3          	vaddss %xmm3,%xmm11,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    436c:	c5 a2 5c db          	vsubss %xmm3,%xmm11,%xmm3
                    x[q + s * (p + m)] = a - b;
    4370:	c5 7a 11 61 f4       	vmovss %xmm12,-0xc(%rcx)
                    x[(q + 1) + s * (p + 0)] = c + d;
    4375:	c5 fa 11 66 f8       	vmovss %xmm4,-0x8(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    437a:	c4 c1 32 58 e2       	vaddss %xmm10,%xmm9,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    437f:	c4 41 32 5c ca       	vsubss %xmm10,%xmm9,%xmm9
                    x[(q + 1) + s * (p + 0)] = c + d;
    4384:	c5 fa 11 66 fc       	vmovss %xmm4,-0x4(%rsi)
                    x[(q + 1) + s * (p + m)] = c - d;
    4389:	c5 fa 11 59 f8       	vmovss %xmm3,-0x8(%rcx)
    438e:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4392:	c5 c2 59 db          	vmulss %xmm3,%xmm7,%xmm3
                    x[(q + 1) + s * (p + m)] = c - d;
    4396:	c5 7a 11 49 fc       	vmovss %xmm9,-0x4(%rcx)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    439b:	c5 fa 59 c6          	vmulss %xmm6,%xmm0,%xmm0
    439f:	c4 e2 41 b9 c1       	vfmadd231ss %xmm1,%xmm7,%xmm0
    43a4:	c4 e2 61 9b ce       	vfmsub132ss %xmm6,%xmm3,%xmm1
        for (int p = 0; p < m; p++)
    43a9:	41 39 fb             	cmp    %edi,%r11d
    43ac:	0f 8f 6e fd ff ff    	jg     4120 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x6e0>
    const float angle = 2 * M_PI / n;
    43b2:	c4 c1 3b 2a c1       	vcvtsi2sd %r9d,%xmm8,%xmm0
    43b7:	89 45 98             	mov    %eax,-0x68(%rbp)
    43ba:	44 89 4d a0          	mov    %r9d,-0x60(%rbp)
    43be:	c5 fa 11 55 90       	vmovss %xmm2,-0x70(%rbp)
    43c3:	c5 fb 11 6d a8       	vmovsd %xmm5,-0x58(%rbp)
    43c8:	c5 d3 5e c0          	vdivsd %xmm0,%xmm5,%xmm0
    43cc:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  { return __builtin_cosf(__x); }
    43d0:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    43d4:	c5 fa 11 4d b0       	vmovss %xmm1,-0x50(%rbp)
    43d9:	e8 22 dd ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    43de:	c5 fa 10 4d b0       	vmovss -0x50(%rbp),%xmm1
    43e3:	c5 fa 11 45 bc       	vmovss %xmm0,-0x44(%rbp)
    43e8:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    43ed:	e8 2e dd ff ff       	call   2120 <sinf@plt>
    43f2:	44 8b 4d a0          	mov    -0x60(%rbp),%r9d
    43f6:	8b 45 98             	mov    -0x68(%rbp),%eax
    43f9:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    43fe:	c5 7a 10 4d bc       	vmovss -0x44(%rbp),%xmm9
    4403:	c5 fb 10 6d a8       	vmovsd -0x58(%rbp),%xmm5
    4408:	c5 78 28 d0          	vmovaps %xmm0,%xmm10
        for (int p = 0; p < m; p++)
    440c:	c5 fa 10 55 90       	vmovss -0x70(%rbp),%xmm2
    4411:	e9 68 fa ff ff       	jmp    3e7e <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x43e>
    const int m = n / 2;
    4416:	85 ff                	test   %edi,%edi
    4418:	8d 57 1f             	lea    0x1f(%rdi),%edx
    441b:	0f 49 d7             	cmovns %edi,%edx
    441e:	c1 fa 05             	sar    $0x5,%edx
    4421:	89 55 bc             	mov    %edx,-0x44(%rbp)
    if (n == 1)
    4424:	83 fa 01             	cmp    $0x1,%edx
    4427:	0f 85 f5 03 00 00    	jne    4822 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xde2>
    442d:	31 d2                	xor    %edx,%edx
    442f:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    4436:	00 00 00 00 
    443a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
                x[q] = y[q];
    4440:	48 8b 0c 13          	mov    (%rbx,%rdx,1),%rcx
    4444:	49 89 0c 17          	mov    %rcx,(%r15,%rdx,1)
            for (int q = 0; q < s; q++)
    4448:	48 83 c2 08          	add    $0x8,%rdx
    444c:	48 81 fa 00 01 00 00 	cmp    $0x100,%rdx
    4453:	75 eb                	jne    4440 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xa00>
    4455:	c5 fa 10 15 93 1c 00 	vmovss 0x1c93(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    445c:	00 
    445d:	c5 fb 10 2d 7b 1c 00 	vmovsd 0x1c7b(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    4464:	00 
    4465:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
    const float angle = 2 * M_PI / n;
    446a:	c4 c1 3b 2a c3       	vcvtsi2sd %r11d,%xmm8,%xmm0
    446f:	89 45 90             	mov    %eax,-0x70(%rbp)
    4472:	44 89 4d 98          	mov    %r9d,-0x68(%rbp)
    4476:	44 89 5d a0          	mov    %r11d,-0x60(%rbp)
    447a:	c5 fb 11 6d a8       	vmovsd %xmm5,-0x58(%rbp)
    447f:	c5 d3 5e c0          	vdivsd %xmm0,%xmm5,%xmm0
    4483:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  { return __builtin_cosf(__x); }
    4487:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    448b:	c5 fa 11 4d b0       	vmovss %xmm1,-0x50(%rbp)
    4490:	e8 6b dc ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    4495:	c5 fa 10 4d b0       	vmovss -0x50(%rbp),%xmm1
    449a:	c5 fa 11 45 88       	vmovss %xmm0,-0x78(%rbp)
    449f:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    44a4:	e8 77 dc ff ff       	call   2120 <sinf@plt>
        for (int p = 0; p < m; p++)
    44a9:	41 83 fd 1f          	cmp    $0x1f,%r13d
    44ad:	44 8b 5d a0          	mov    -0x60(%rbp),%r11d
    44b1:	8b 45 90             	mov    -0x70(%rbp),%eax
    44b4:	c5 fb 10 6d a8       	vmovsd -0x58(%rbp),%xmm5
    44b9:	44 8b 4d 98          	mov    -0x68(%rbp),%r9d
    44bd:	c5 78 28 d0          	vmovaps %xmm0,%xmm10
    44c1:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    44c6:	0f 8e c0 fb ff ff    	jle    408c <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x64c>
    44cc:	c5 fa 10 15 38 1b 00 	vmovss 0x1b38(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    44d3:	00 
    44d4:	c5 7a 10 4d 88       	vmovss -0x78(%rbp),%xmm9
    44d9:	48 63 55 bc          	movslq -0x44(%rbp),%rdx
    44dd:	4c 89 6d a0          	mov    %r13,-0x60(%rbp)
    44e1:	4c 89 fe             	mov    %r15,%rsi
    44e4:	45 31 d2             	xor    %r10d,%r10d
    44e7:	4c 89 75 98          	mov    %r14,-0x68(%rbp)
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    44eb:	c5 f8 28 ca          	vmovaps %xmm2,%xmm1
    44ef:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    44f3:	48 89 d1             	mov    %rdx,%rcx
    44f6:	48 c1 e2 07          	shl    $0x7,%rdx
    44fa:	89 45 90             	mov    %eax,-0x70(%rbp)
    44fd:	4c 8d 04 13          	lea    (%rbx,%rdx,1),%r8
    4501:	48 29 da             	sub    %rbx,%rdx
    4504:	89 cf                	mov    %ecx,%edi
    4506:	4c 89 7d a8          	mov    %r15,-0x58(%rbp)
    450a:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
    450e:	ba 20 00 00 00       	mov    $0x20,%edx
    4513:	c1 e7 04             	shl    $0x4,%edi
    4516:	4c 8b 75 b0          	mov    -0x50(%rbp),%r14
    451a:	48 29 da             	sub    %rbx,%rdx
    451d:	89 7d bc             	mov    %edi,-0x44(%rbp)
    4520:	89 c8                	mov    %ecx,%eax
    4522:	48 89 df             	mov    %rbx,%rdi
    4525:	44 89 65 b0          	mov    %r12d,-0x50(%rbp)
    4529:	49 89 d5             	mov    %rdx,%r13
    452c:	44 8b 65 bc          	mov    -0x44(%rbp),%r12d
    4530:	44 89 4d bc          	mov    %r9d,-0x44(%rbp)
    4534:	eb 0e                	jmp    4544 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xb04>
    4536:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    453d:	00 00 00 
    4540:	c5 f8 28 c3          	vmovaps %xmm3,%xmm0
                for (int q = 0; q < s; q += 2)
    4544:	4c 89 d1             	mov    %r10,%rcx
    4547:	4f 8d 0c 28          	lea    (%r8,%r13,1),%r9
    454b:	4e 8d 3c 37          	lea    (%rdi,%r14,1),%r15
    454f:	48 c1 e1 07          	shl    $0x7,%rcx
    4553:	48 8d 56 10          	lea    0x10(%rsi),%rdx
    4557:	49 39 c9             	cmp    %rcx,%r9
    455a:	4e 8d 0c 2f          	lea    (%rdi,%r13,1),%r9
    455e:	0f 9e c1             	setle  %cl
    4561:	4d 39 cf             	cmp    %r9,%r15
    4564:	41 0f 9d c1          	setge  %r9b
    4568:	41 08 c9             	or     %cl,%r9b
    456b:	0f 84 df 01 00 00    	je     4750 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xd10>
    4571:	48 89 f9             	mov    %rdi,%rcx
    4574:	48 29 d1             	sub    %rdx,%rcx
    4577:	48 83 c1 0c          	add    $0xc,%rcx
    457b:	48 81 f9 98 00 00 00 	cmp    $0x98,%rcx
    4582:	4c 89 c1             	mov    %r8,%rcx
    4585:	41 0f 97 c1          	seta   %r9b
    4589:	48 29 d1             	sub    %rdx,%rcx
    458c:	48 83 c1 0c          	add    $0xc,%rcx
    4590:	48 81 f9 98 00 00 00 	cmp    $0x98,%rcx
    4597:	0f 97 c2             	seta   %dl
    459a:	41 84 d1             	test   %dl,%r9b
    459d:	0f 84 ad 01 00 00    	je     4750 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xd10>
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    45a3:	c5 fc 10 b6 80 00 00 	vmovups 0x80(%rsi),%ymm6
    45aa:	00 
    45ab:	c5 f8 14 e1          	vunpcklps %xmm1,%xmm0,%xmm4
    45af:	c5 f0 14 d8          	vunpcklps %xmm0,%xmm1,%xmm3
                    const complex_t a = y[q + s * (2 * p + 0)];
    45b3:	c5 fc 10 3e          	vmovups (%rsi),%ymm7
    45b7:	c5 d8 16 e4          	vmovlhps %xmm4,%xmm4,%xmm4
    45bb:	c5 e0 16 db          	vmovlhps %xmm3,%xmm3,%xmm3
    45bf:	c4 e3 5d 18 e4 01    	vinsertf128 $0x1,%xmm4,%ymm4,%ymm4
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    45c5:	c4 63 7d 04 de a0    	vpermilps $0xa0,%ymm6,%ymm11
    45cb:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    45d1:	c5 cc 59 f4          	vmulps %ymm4,%ymm6,%ymm6
    45d5:	c4 e3 65 18 db 01    	vinsertf128 $0x1,%xmm3,%ymm3,%ymm3
    45db:	c4 e2 25 b6 f3       	vfmaddsub231ps %ymm3,%ymm11,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    45e0:	c5 44 58 de          	vaddps %ymm6,%ymm7,%ymm11
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    45e4:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    45e8:	c5 7c 11 1f          	vmovups %ymm11,(%rdi)
                    x[q + s * (p + m)] = a - b;
    45ec:	c4 c1 7c 11 38       	vmovups %ymm7,(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    45f1:	c5 fc 10 b6 a0 00 00 	vmovups 0xa0(%rsi),%ymm6
    45f8:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    45f9:	c5 fc 10 7e 20       	vmovups 0x20(%rsi),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    45fe:	c4 63 7d 04 de a0    	vpermilps $0xa0,%ymm6,%ymm11
    4604:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    460a:	c5 cc 59 f4          	vmulps %ymm4,%ymm6,%ymm6
    460e:	c4 e2 25 b6 f3       	vfmaddsub231ps %ymm3,%ymm11,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4613:	c5 44 58 de          	vaddps %ymm6,%ymm7,%ymm11
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4617:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    461b:	c5 7c 11 5f 20       	vmovups %ymm11,0x20(%rdi)
                    x[q + s * (p + m)] = a - b;
    4620:	c4 c1 7c 11 78 20    	vmovups %ymm7,0x20(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4626:	c5 fc 10 b6 c0 00 00 	vmovups 0xc0(%rsi),%ymm6
    462d:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    462e:	c5 fc 10 7e 40       	vmovups 0x40(%rsi),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4633:	c4 63 7d 04 de a0    	vpermilps $0xa0,%ymm6,%ymm11
    4639:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    463f:	c5 cc 59 f4          	vmulps %ymm4,%ymm6,%ymm6
    4643:	c4 e2 25 b6 f3       	vfmaddsub231ps %ymm3,%ymm11,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4648:	c5 4c 58 df          	vaddps %ymm7,%ymm6,%ymm11
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    464c:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4650:	c5 7c 11 5f 40       	vmovups %ymm11,0x40(%rdi)
                    x[q + s * (p + m)] = a - b;
    4655:	c4 c1 7c 11 78 40    	vmovups %ymm7,0x40(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    465b:	c5 fc 10 b6 e0 00 00 	vmovups 0xe0(%rsi),%ymm6
    4662:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4663:	c5 fc 10 7e 60       	vmovups 0x60(%rsi),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4668:	c4 63 7d 04 de a0    	vpermilps $0xa0,%ymm6,%ymm11
    466e:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4674:	c5 cc 59 f4          	vmulps %ymm4,%ymm6,%ymm6
    4678:	c4 c2 4d 96 db       	vfmaddsub132ps %ymm11,%ymm6,%ymm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    467d:	c5 e4 58 e7          	vaddps %ymm7,%ymm3,%ymm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4681:	c5 c4 5c fb          	vsubps %ymm3,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4685:	c5 fc 11 67 60       	vmovups %ymm4,0x60(%rdi)
                    x[q + s * (p + m)] = a - b;
    468a:	c4 c1 7c 11 78 60    	vmovups %ymm7,0x60(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4690:	c4 c1 7a 59 d9       	vmulss %xmm9,%xmm0,%xmm3
        for (int p = 0; p < m; p++)
    4695:	49 ff c2             	inc    %r10
    4698:	49 83 e8 80          	sub    $0xffffffffffffff80,%r8
    469c:	48 83 ef 80          	sub    $0xffffffffffffff80,%rdi
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    46a0:	c4 c1 7a 59 c2       	vmulss %xmm10,%xmm0,%xmm0
        for (int p = 0; p < m; p++)
    46a5:	48 81 c6 00 01 00 00 	add    $0x100,%rsi
    46ac:	41 83 c4 10          	add    $0x10,%r12d
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    46b0:	c4 c2 71 b9 da       	vfmadd231ss %xmm10,%xmm1,%xmm3
    46b5:	c4 c2 79 9b c9       	vfmsub132ss %xmm9,%xmm0,%xmm1
        for (int p = 0; p < m; p++)
    46ba:	44 39 d0             	cmp    %r10d,%eax
    46bd:	0f 8f 7d fe ff ff    	jg     4540 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xb00>
    const float angle = 2 * M_PI / n;
    46c3:	8b 45 90             	mov    -0x70(%rbp),%eax
    46c6:	44 8b 4d bc          	mov    -0x44(%rbp),%r9d
    46ca:	44 89 5d 90          	mov    %r11d,-0x70(%rbp)
    46ce:	4c 8b 6d a0          	mov    -0x60(%rbp),%r13
    46d2:	4c 8b 7d a8          	mov    -0x58(%rbp),%r15
    46d6:	c5 fb 11 6d a8       	vmovsd %xmm5,-0x58(%rbp)
    46db:	c5 bb 2a c0          	vcvtsi2sd %eax,%xmm8,%xmm0
    46df:	4c 8b 75 98          	mov    -0x68(%rbp),%r14
    46e3:	44 8b 65 b0          	mov    -0x50(%rbp),%r12d
    46e7:	44 89 4d 88          	mov    %r9d,-0x78(%rbp)
    46eb:	89 45 a0             	mov    %eax,-0x60(%rbp)
    46ee:	c5 fa 11 55 98       	vmovss %xmm2,-0x68(%rbp)
    46f3:	c5 d3 5e c0          	vdivsd %xmm0,%xmm5,%xmm0
    46f7:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  { return __builtin_cosf(__x); }
    46fb:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    46ff:	c5 fa 11 4d b0       	vmovss %xmm1,-0x50(%rbp)
    4704:	c5 f8 77             	vzeroupper
    4707:	e8 f4 d9 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    470c:	c5 fa 10 4d b0       	vmovss -0x50(%rbp),%xmm1
    4711:	c5 fa 11 45 bc       	vmovss %xmm0,-0x44(%rbp)
    4716:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    471b:	e8 00 da ff ff       	call   2120 <sinf@plt>
    4720:	8b 45 a0             	mov    -0x60(%rbp),%eax
    4723:	44 8b 5d 90          	mov    -0x70(%rbp),%r11d
    4727:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    472c:	c5 fa 10 75 bc       	vmovss -0x44(%rbp),%xmm6
    4731:	c5 fb 10 6d a8       	vmovsd -0x58(%rbp),%xmm5
    4736:	c5 f8 28 f8          	vmovaps %xmm0,%xmm7
        for (int p = 0; p < m; p++)
    473a:	c5 fa 10 55 98       	vmovss -0x68(%rbp),%xmm2
    473f:	44 8b 4d 88          	mov    -0x78(%rbp),%r9d
    4743:	e9 b2 f9 ff ff       	jmp    40fa <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x6ba>
    4748:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    474f:	00 
    4750:	49 63 d4             	movslq %r12d,%rdx
                    x[q + s * (p + m)] = a - b;
    4753:	48 89 f9             	mov    %rdi,%rcx
    4756:	4c 8d be 80 00 00 00 	lea    0x80(%rsi),%r15
    475d:	4c 8d 0c d3          	lea    (%rbx,%rdx,8),%r9
    4761:	48 89 f2             	mov    %rsi,%rdx
    4764:	0f 1f 40 00          	nopl   0x0(%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4768:	c5 fa 10 9a 84 00 00 	vmovss 0x84(%rdx),%xmm3
    476f:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4770:	c5 7a 10 2a          	vmovss (%rdx),%xmm13
                for (int q = 0; q < s; q += 2)
    4774:	48 83 c2 10          	add    $0x10,%rdx
    4778:	48 83 c1 10          	add    $0x10,%rcx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    477c:	c5 fa 10 62 70       	vmovss 0x70(%rdx),%xmm4
    4781:	c5 7a 10 72 7c       	vmovss 0x7c(%rdx),%xmm14
                for (int q = 0; q < s; q += 2)
    4786:	49 83 c1 10          	add    $0x10,%r9
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    478a:	c5 72 59 e3          	vmulss %xmm3,%xmm1,%xmm12
                    const complex_t a = y[q + s * (2 * p + 0)];
    478e:	c5 7a 10 5a f4       	vmovss -0xc(%rdx),%xmm11
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    4793:	c5 7a 10 7a f8       	vmovss -0x8(%rdx),%xmm15
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4798:	c5 fa 59 db          	vmulss %xmm3,%xmm0,%xmm3
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    479c:	c5 fa 10 72 fc       	vmovss -0x4(%rdx),%xmm6
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    47a1:	c4 c1 72 59 fe       	vmulss %xmm14,%xmm1,%xmm7
    47a6:	c4 41 7a 59 f6       	vmulss %xmm14,%xmm0,%xmm14
    47ab:	c4 62 79 b9 e4       	vfmadd231ss %xmm4,%xmm0,%xmm12
    47b0:	c4 e2 61 9b e1       	vfmsub132ss %xmm1,%xmm3,%xmm4
    47b5:	c5 fa 10 5a 78       	vmovss 0x78(%rdx),%xmm3
    47ba:	c4 e2 79 b9 fb       	vfmadd231ss %xmm3,%xmm0,%xmm7
    47bf:	c4 e2 09 9b d9       	vfmsub132ss %xmm1,%xmm14,%xmm3
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    47c4:	c4 41 5a 58 f5       	vaddss %xmm13,%xmm4,%xmm14
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    47c9:	c5 92 5c e4          	vsubss %xmm4,%xmm13,%xmm4
                    x[q + s * (p + 0)] = a + b;
    47cd:	c5 7a 11 71 f0       	vmovss %xmm14,-0x10(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    47d2:	c4 41 1a 58 f3       	vaddss %xmm11,%xmm12,%xmm14
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    47d7:	c4 41 22 5c dc       	vsubss %xmm12,%xmm11,%xmm11
                    x[q + s * (p + 0)] = a + b;
    47dc:	c5 7a 11 71 f4       	vmovss %xmm14,-0xc(%rcx)
                    x[q + s * (p + m)] = a - b;
    47e1:	c4 c1 7a 11 61 f0    	vmovss %xmm4,-0x10(%r9)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    47e7:	c4 c1 62 58 e7       	vaddss %xmm15,%xmm3,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    47ec:	c5 82 5c db          	vsubss %xmm3,%xmm15,%xmm3
                    x[q + s * (p + m)] = a - b;
    47f0:	c4 41 7a 11 59 f4    	vmovss %xmm11,-0xc(%r9)
                    x[(q + 1) + s * (p + 0)] = c + d;
    47f6:	c5 fa 11 61 f8       	vmovss %xmm4,-0x8(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    47fb:	c5 c2 58 e6          	vaddss %xmm6,%xmm7,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    47ff:	c5 ca 5c f7          	vsubss %xmm7,%xmm6,%xmm6
                    x[(q + 1) + s * (p + 0)] = c + d;
    4803:	c5 fa 11 61 fc       	vmovss %xmm4,-0x4(%rcx)
                    x[(q + 1) + s * (p + m)] = c - d;
    4808:	c4 c1 7a 11 59 f8    	vmovss %xmm3,-0x8(%r9)
    480e:	c4 c1 7a 11 71 fc    	vmovss %xmm6,-0x4(%r9)
                for (int q = 0; q < s; q += 2)
    4814:	4c 39 fa             	cmp    %r15,%rdx
    4817:	0f 85 4b ff ff ff    	jne    4768 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xd28>
    481d:	e9 6e fe ff ff       	jmp    4690 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xc50>
    const int m = n / 2;
    4822:	85 ff                	test   %edi,%edi
    4824:	8d 57 3f             	lea    0x3f(%rdi),%edx
    4827:	0f 49 d7             	cmovns %edi,%edx
    482a:	c1 fa 06             	sar    $0x6,%edx
    482d:	89 55 a8             	mov    %edx,-0x58(%rbp)
    if (n == 1)
    4830:	83 fa 01             	cmp    $0x1,%edx
    4833:	0f 85 ce 04 00 00    	jne    4d07 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x12c7>
    4839:	c5 fa 10 15 af 18 00 	vmovss 0x18af(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    4840:	00 
    4841:	c5 fb 10 2d 97 18 00 	vmovsd 0x1897(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    4848:	00 
    4849:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
    const float angle = 2 * M_PI / n;
    484e:	c5 bb 2a 4d bc       	vcvtsi2sdl -0x44(%rbp),%xmm8,%xmm1
    4853:	44 89 5d 80          	mov    %r11d,-0x80(%rbp)
    4857:	89 45 88             	mov    %eax,-0x78(%rbp)
    485a:	44 89 4d 90          	mov    %r9d,-0x70(%rbp)
    485e:	c5 fb 11 6d 98       	vmovsd %xmm5,-0x68(%rbp)
    4863:	c5 d3 5e c9          	vdivsd %xmm1,%xmm5,%xmm1
    4867:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
  { return __builtin_cosf(__x); }
    486b:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    486f:	c5 fa 11 4d a0       	vmovss %xmm1,-0x60(%rbp)
    4874:	e8 87 d8 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    4879:	c5 fa 10 4d a0       	vmovss -0x60(%rbp),%xmm1
    487e:	c5 fa 11 45 b0       	vmovss %xmm0,-0x50(%rbp)
    4883:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    4888:	e8 93 d8 ff ff       	call   2120 <sinf@plt>
        for (int p = 0; p < m; p++)
    488d:	41 83 fd 3f          	cmp    $0x3f,%r13d
    4891:	44 8b 4d 90          	mov    -0x70(%rbp),%r9d
    4895:	8b 45 88             	mov    -0x78(%rbp),%eax
    4898:	c5 fb 10 6d 98       	vmovsd -0x68(%rbp),%xmm5
    489d:	44 8b 5d 80          	mov    -0x80(%rbp),%r11d
    48a1:	c5 78 28 d8          	vmovaps %xmm0,%xmm11
    48a5:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    48aa:	0f 8e ba fb ff ff    	jle    446a <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xa2a>
    48b0:	c5 fa 10 15 54 17 00 	vmovss 0x1754(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    48b7:	00 
    48b8:	c5 7a 10 55 b0       	vmovss -0x50(%rbp),%xmm10
    48bd:	48 63 55 a8          	movslq -0x58(%rbp),%rdx
    48c1:	48 89 5d 98          	mov    %rbx,-0x68(%rbp)
    48c5:	4c 89 ff             	mov    %r15,%rdi
    48c8:	45 31 d2             	xor    %r10d,%r10d
    48cb:	48 89 5d a8          	mov    %rbx,-0x58(%rbp)
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    48cf:	c5 f8 28 e2          	vmovaps %xmm2,%xmm4
    48d3:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    48d7:	48 89 d6             	mov    %rdx,%rsi
    48da:	48 c1 e2 08          	shl    $0x8,%rdx
    48de:	44 89 65 88          	mov    %r12d,-0x78(%rbp)
    48e2:	89 f1                	mov    %esi,%ecx
    48e4:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
    48e8:	4d 8d 04 17          	lea    (%r15,%rdx,1),%r8
    48ec:	48 83 c2 20          	add    $0x20,%rdx
    48f0:	c1 e1 05             	shl    $0x5,%ecx
    48f3:	48 89 55 a0          	mov    %rdx,-0x60(%rbp)
    48f7:	4c 8b 65 b0          	mov    -0x50(%rbp),%r12
    48fb:	44 89 5d b8          	mov    %r11d,-0x48(%rbp)
    48ff:	48 8b 55 a8          	mov    -0x58(%rbp),%rdx
    4903:	89 cb                	mov    %ecx,%ebx
    4905:	41 89 f3             	mov    %esi,%r11d
    4908:	4c 89 6d a8          	mov    %r13,-0x58(%rbp)
    490c:	4c 8b 6d a0          	mov    -0x60(%rbp),%r13
    4910:	44 89 4d b0          	mov    %r9d,-0x50(%rbp)
    4914:	45 31 c9             	xor    %r9d,%r9d
    4917:	4c 89 75 90          	mov    %r14,-0x70(%rbp)
    491b:	89 45 80             	mov    %eax,-0x80(%rbp)
    491e:	eb 04                	jmp    4924 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xee4>
    4920:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
                for (int q = 0; q < s; q += 2)
    4924:	4b 8d 4c 15 00       	lea    0x0(%r13,%r10,1),%rcx
    4929:	4b 8d 34 14          	lea    (%r12,%r10,1),%rsi
    492d:	49 39 ca             	cmp    %rcx,%r10
    4930:	4d 8d 72 20          	lea    0x20(%r10),%r14
    4934:	48 8d 42 10          	lea    0x10(%rdx),%rax
    4938:	0f 9d c1             	setge  %cl
    493b:	49 39 f6             	cmp    %rsi,%r14
    493e:	40 0f 9e c6          	setle  %sil
    4942:	40 08 ce             	or     %cl,%sil
    4945:	0f 84 e6 02 00 00    	je     4c31 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x11f1>
    494b:	48 89 f9             	mov    %rdi,%rcx
    494e:	48 29 c1             	sub    %rax,%rcx
    4951:	48 83 c1 0c          	add    $0xc,%rcx
    4955:	48 81 f9 18 01 00 00 	cmp    $0x118,%rcx
    495c:	4c 89 c1             	mov    %r8,%rcx
    495f:	40 0f 97 c6          	seta   %sil
    4963:	48 29 c1             	sub    %rax,%rcx
    4966:	48 83 c1 0c          	add    $0xc,%rcx
    496a:	48 81 f9 18 01 00 00 	cmp    $0x118,%rcx
    4971:	0f 97 c0             	seta   %al
    4974:	40 84 c6             	test   %al,%sil
    4977:	0f 84 b4 02 00 00    	je     4c31 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x11f1>
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    497d:	c5 fc 10 b2 00 01 00 	vmovups 0x100(%rdx),%ymm6
    4984:	00 
    4985:	c5 e0 14 cc          	vunpcklps %xmm4,%xmm3,%xmm1
    4989:	c5 d8 14 c3          	vunpcklps %xmm3,%xmm4,%xmm0
                    const complex_t a = y[q + s * (2 * p + 0)];
    498d:	c5 fc 10 3a          	vmovups (%rdx),%ymm7
    4991:	c5 f0 16 c9          	vmovlhps %xmm1,%xmm1,%xmm1
    4995:	c5 f8 16 c0          	vmovlhps %xmm0,%xmm0,%xmm0
    4999:	c4 e3 75 18 c9 01    	vinsertf128 $0x1,%xmm1,%ymm1,%ymm1
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    499f:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    49a5:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    49ab:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    49af:	c4 e3 7d 18 c0 01    	vinsertf128 $0x1,%xmm0,%ymm0,%ymm0
    49b5:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    49ba:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    49be:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    49c2:	c5 7c 11 0f          	vmovups %ymm9,(%rdi)
                    x[q + s * (p + m)] = a - b;
    49c6:	c4 c1 7c 11 38       	vmovups %ymm7,(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    49cb:	c5 fc 10 b2 20 01 00 	vmovups 0x120(%rdx),%ymm6
    49d2:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    49d3:	c5 fc 10 7a 20       	vmovups 0x20(%rdx),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    49d8:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    49de:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    49e4:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    49e8:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    49ed:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    49f1:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    49f5:	c5 7c 11 4f 20       	vmovups %ymm9,0x20(%rdi)
                    x[q + s * (p + m)] = a - b;
    49fa:	c4 c1 7c 11 78 20    	vmovups %ymm7,0x20(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4a00:	c5 fc 10 b2 40 01 00 	vmovups 0x140(%rdx),%ymm6
    4a07:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4a08:	c5 fc 10 7a 40       	vmovups 0x40(%rdx),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4a0d:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4a13:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4a19:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4a1d:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4a22:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4a26:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4a2a:	c5 7c 11 4f 40       	vmovups %ymm9,0x40(%rdi)
                    x[q + s * (p + m)] = a - b;
    4a2f:	c4 c1 7c 11 78 40    	vmovups %ymm7,0x40(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4a35:	c5 fc 10 b2 60 01 00 	vmovups 0x160(%rdx),%ymm6
    4a3c:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4a3d:	c5 fc 10 7a 60       	vmovups 0x60(%rdx),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4a42:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4a48:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4a4e:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4a52:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4a57:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4a5b:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4a5f:	c5 7c 11 4f 60       	vmovups %ymm9,0x60(%rdi)
                    x[q + s * (p + m)] = a - b;
    4a64:	c4 c1 7c 11 78 60    	vmovups %ymm7,0x60(%r8)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4a6a:	c5 fc 10 b2 80 01 00 	vmovups 0x180(%rdx),%ymm6
    4a71:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4a72:	c5 fc 10 ba 80 00 00 	vmovups 0x80(%rdx),%ymm7
    4a79:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4a7a:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4a80:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4a86:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4a8a:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4a8f:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4a93:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4a97:	c5 7c 11 8f 80 00 00 	vmovups %ymm9,0x80(%rdi)
    4a9e:	00 
                    x[q + s * (p + m)] = a - b;
    4a9f:	c4 c1 7c 11 b8 80 00 	vmovups %ymm7,0x80(%r8)
    4aa6:	00 00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4aa8:	c5 fc 10 b2 a0 01 00 	vmovups 0x1a0(%rdx),%ymm6
    4aaf:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4ab0:	c5 fc 10 ba a0 00 00 	vmovups 0xa0(%rdx),%ymm7
    4ab7:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4ab8:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4abe:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4ac4:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4ac8:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4acd:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4ad1:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4ad5:	c5 7c 11 8f a0 00 00 	vmovups %ymm9,0xa0(%rdi)
    4adc:	00 
                    x[q + s * (p + m)] = a - b;
    4add:	c4 c1 7c 11 b8 a0 00 	vmovups %ymm7,0xa0(%r8)
    4ae4:	00 00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4ae6:	c5 fc 10 b2 c0 01 00 	vmovups 0x1c0(%rdx),%ymm6
    4aed:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4aee:	c5 fc 10 ba c0 00 00 	vmovups 0xc0(%rdx),%ymm7
    4af5:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4af6:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4afc:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4b02:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4b06:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4b0b:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4b0f:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4b13:	c5 7c 11 8f c0 00 00 	vmovups %ymm9,0xc0(%rdi)
    4b1a:	00 
                    x[q + s * (p + m)] = a - b;
    4b1b:	c4 c1 7c 11 b8 c0 00 	vmovups %ymm7,0xc0(%r8)
    4b22:	00 00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4b24:	c5 fc 10 b2 e0 01 00 	vmovups 0x1e0(%rdx),%ymm6
    4b2b:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4b2c:	c5 fc 10 ba e0 00 00 	vmovups 0xe0(%rdx),%ymm7
    4b33:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4b34:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4b3a:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4b40:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4b44:	c4 c2 4d 96 c1       	vfmaddsub132ps %ymm9,%ymm6,%ymm0
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4b49:	c5 fc 58 cf          	vaddps %ymm7,%ymm0,%ymm1
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4b4d:	c5 c4 5c f8          	vsubps %ymm0,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4b51:	c5 fc 11 8f e0 00 00 	vmovups %ymm1,0xe0(%rdi)
    4b58:	00 
                    x[q + s * (p + m)] = a - b;
    4b59:	c4 c1 7c 11 b8 e0 00 	vmovups %ymm7,0xe0(%r8)
    4b60:	00 00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4b62:	c4 c1 62 59 c2       	vmulss %xmm10,%xmm3,%xmm0
        for (int p = 0; p < m; p++)
    4b67:	41 ff c1             	inc    %r9d
    4b6a:	49 81 c2 00 01 00 00 	add    $0x100,%r10
    4b71:	49 81 c0 00 01 00 00 	add    $0x100,%r8
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4b78:	c4 c1 62 59 db       	vmulss %xmm11,%xmm3,%xmm3
        for (int p = 0; p < m; p++)
    4b7d:	48 81 c2 00 02 00 00 	add    $0x200,%rdx
    4b84:	48 81 c7 00 01 00 00 	add    $0x100,%rdi
    4b8b:	83 c3 20             	add    $0x20,%ebx
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4b8e:	c4 c2 59 b9 c3       	vfmadd231ss %xmm11,%xmm4,%xmm0
    4b93:	c4 c2 61 9b e2       	vfmsub132ss %xmm10,%xmm3,%xmm4
        for (int p = 0; p < m; p++)
    4b98:	45 39 cb             	cmp    %r9d,%r11d
    4b9b:	0f 8f 7f fd ff ff    	jg     4920 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xee0>
    const float angle = 2 * M_PI / n;
    4ba1:	44 8b 5d b8          	mov    -0x48(%rbp),%r11d
    4ba5:	44 8b 4d b0          	mov    -0x50(%rbp),%r9d
    4ba9:	c5 fb 11 6d a0       	vmovsd %xmm5,-0x60(%rbp)
    4bae:	8b 45 80             	mov    -0x80(%rbp),%eax
    4bb1:	48 8b 5d 98          	mov    -0x68(%rbp),%rbx
    4bb5:	c4 c1 3b 2a c3       	vcvtsi2sd %r11d,%xmm8,%xmm0
    4bba:	44 8b 65 88          	mov    -0x78(%rbp),%r12d
    4bbe:	4c 8b 6d a8          	mov    -0x58(%rbp),%r13
    4bc2:	44 89 5d 98          	mov    %r11d,-0x68(%rbp)
    4bc6:	44 89 8d 7c ff ff ff 	mov    %r9d,-0x84(%rbp)
    4bcd:	4c 8b 75 90          	mov    -0x70(%rbp),%r14
    4bd1:	89 45 88             	mov    %eax,-0x78(%rbp)
    4bd4:	c5 fa 11 55 90       	vmovss %xmm2,-0x70(%rbp)
    4bd9:	c5 d3 5e c0          	vdivsd %xmm0,%xmm5,%xmm0
    4bdd:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  { return __builtin_cosf(__x); }
    4be1:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    4be5:	c5 fa 11 4d a8       	vmovss %xmm1,-0x58(%rbp)
    4bea:	c5 f8 77             	vzeroupper
    4bed:	e8 0e d5 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    4bf2:	c5 fa 10 4d a8       	vmovss -0x58(%rbp),%xmm1
    4bf7:	c5 fa 11 45 b0       	vmovss %xmm0,-0x50(%rbp)
    4bfc:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    4c01:	e8 1a d5 ff ff       	call   2120 <sinf@plt>
    4c06:	44 8b 5d 98          	mov    -0x68(%rbp),%r11d
    4c0a:	8b 45 88             	mov    -0x78(%rbp),%eax
    4c0d:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    4c12:	c5 7a 10 4d b0       	vmovss -0x50(%rbp),%xmm9
    4c17:	c5 fb 10 6d a0       	vmovsd -0x60(%rbp),%xmm5
    4c1c:	c5 78 28 d0          	vmovaps %xmm0,%xmm10
        for (int p = 0; p < m; p++)
    4c20:	c5 fa 10 55 90       	vmovss -0x70(%rbp),%xmm2
    4c25:	44 8b 8d 7c ff ff ff 	mov    -0x84(%rbp),%r9d
    4c2c:	e9 a8 f8 ff ff       	jmp    44d9 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xa99>
    4c31:	48 63 c3             	movslq %ebx,%rax
                    x[q + s * (p + m)] = a - b;
    4c34:	48 89 f9             	mov    %rdi,%rcx
    4c37:	4c 8d b2 00 01 00 00 	lea    0x100(%rdx),%r14
    4c3e:	49 8d 34 c7          	lea    (%r15,%rax,8),%rsi
    4c42:	48 89 d0             	mov    %rdx,%rax
    4c45:	0f 1f 00             	nopl   (%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4c48:	c5 fa 10 80 04 01 00 	vmovss 0x104(%rax),%xmm0
    4c4f:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4c50:	c5 7a 10 30          	vmovss (%rax),%xmm14
                for (int q = 0; q < s; q += 2)
    4c54:	48 83 c0 10          	add    $0x10,%rax
    4c58:	48 83 c1 10          	add    $0x10,%rcx
                    const complex_t a = y[q + s * (2 * p + 0)];
    4c5c:	c5 7a 10 60 f4       	vmovss -0xc(%rax),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    4c61:	c5 7a 10 48 f8       	vmovss -0x8(%rax),%xmm9
                for (int q = 0; q < s; q += 2)
    4c66:	48 83 c6 10          	add    $0x10,%rsi
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4c6a:	c5 5a 59 e8          	vmulss %xmm0,%xmm4,%xmm13
    4c6e:	c5 fa 10 88 f0 00 00 	vmovss 0xf0(%rax),%xmm1
    4c75:	00 
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    4c76:	c5 fa 10 70 fc       	vmovss -0x4(%rax),%xmm6
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4c7b:	c5 e2 59 c0          	vmulss %xmm0,%xmm3,%xmm0
    4c7f:	c5 7a 10 b8 fc 00 00 	vmovss 0xfc(%rax),%xmm15
    4c86:	00 
    4c87:	c4 c1 5a 59 ff       	vmulss %xmm15,%xmm4,%xmm7
    4c8c:	c4 41 62 59 ff       	vmulss %xmm15,%xmm3,%xmm15
    4c91:	c4 62 61 b9 e9       	vfmadd231ss %xmm1,%xmm3,%xmm13
    4c96:	c4 e2 79 9b cc       	vfmsub132ss %xmm4,%xmm0,%xmm1
    4c9b:	c5 fa 10 80 f8 00 00 	vmovss 0xf8(%rax),%xmm0
    4ca2:	00 
    4ca3:	c4 e2 61 b9 f8       	vfmadd231ss %xmm0,%xmm3,%xmm7
    4ca8:	c4 e2 01 9b c4       	vfmsub132ss %xmm4,%xmm15,%xmm0
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4cad:	c4 41 72 58 fe       	vaddss %xmm14,%xmm1,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4cb2:	c5 8a 5c c9          	vsubss %xmm1,%xmm14,%xmm1
                    x[q + s * (p + 0)] = a + b;
    4cb6:	c5 7a 11 79 f0       	vmovss %xmm15,-0x10(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4cbb:	c4 41 12 58 fc       	vaddss %xmm12,%xmm13,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4cc0:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    4cc5:	c5 7a 11 79 f4       	vmovss %xmm15,-0xc(%rcx)
                    x[q + s * (p + m)] = a - b;
    4cca:	c5 fa 11 4e f0       	vmovss %xmm1,-0x10(%rsi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4ccf:	c4 c1 7a 58 c9       	vaddss %xmm9,%xmm0,%xmm1
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4cd4:	c5 b2 5c c0          	vsubss %xmm0,%xmm9,%xmm0
                    x[q + s * (p + m)] = a - b;
    4cd8:	c5 7a 11 66 f4       	vmovss %xmm12,-0xc(%rsi)
                    x[(q + 1) + s * (p + 0)] = c + d;
    4cdd:	c5 fa 11 49 f8       	vmovss %xmm1,-0x8(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4ce2:	c5 c2 58 ce          	vaddss %xmm6,%xmm7,%xmm1
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4ce6:	c5 ca 5c f7          	vsubss %xmm7,%xmm6,%xmm6
                    x[(q + 1) + s * (p + 0)] = c + d;
    4cea:	c5 fa 11 49 fc       	vmovss %xmm1,-0x4(%rcx)
                    x[(q + 1) + s * (p + m)] = c - d;
    4cef:	c5 fa 11 46 f8       	vmovss %xmm0,-0x8(%rsi)
    4cf4:	c5 fa 11 76 fc       	vmovss %xmm6,-0x4(%rsi)
                for (int q = 0; q < s; q += 2)
    4cf9:	4c 39 f0             	cmp    %r14,%rax
    4cfc:	0f 85 46 ff ff ff    	jne    4c48 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1208>
    4d02:	e9 5b fe ff ff       	jmp    4b62 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1122>
    const int m = n / 2;
    4d07:	85 ff                	test   %edi,%edi
    4d09:	8d 57 7f             	lea    0x7f(%rdi),%edx
    4d0c:	0f 49 d7             	cmovns %edi,%edx
    4d0f:	c1 fa 07             	sar    $0x7,%edx
    4d12:	89 55 98             	mov    %edx,-0x68(%rbp)
    if (n == 1)
    4d15:	83 fa 01             	cmp    $0x1,%edx
    4d18:	0f 85 cd 06 00 00    	jne    53eb <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x19ab>
    4d1e:	31 d2                	xor    %edx,%edx
                x[q] = y[q];
    4d20:	48 8b 0c 13          	mov    (%rbx,%rdx,1),%rcx
    4d24:	49 89 0c 17          	mov    %rcx,(%r15,%rdx,1)
            for (int q = 0; q < s; q++)
    4d28:	48 83 c2 08          	add    $0x8,%rdx
    4d2c:	48 81 fa 00 04 00 00 	cmp    $0x400,%rdx
    4d33:	75 eb                	jne    4d20 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x12e0>
    4d35:	c5 fa 10 15 b3 13 00 	vmovss 0x13b3(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    4d3c:	00 
    4d3d:	c5 fb 10 2d 9b 13 00 	vmovsd 0x139b(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    4d44:	00 
    4d45:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
    const float angle = 2 * M_PI / n;
    4d4a:	c5 bb 2a 4d a8       	vcvtsi2sdl -0x58(%rbp),%xmm8,%xmm1
    4d4f:	44 89 5d b8          	mov    %r11d,-0x48(%rbp)
    4d53:	89 45 80             	mov    %eax,-0x80(%rbp)
    4d56:	44 89 4d 88          	mov    %r9d,-0x78(%rbp)
    4d5a:	c5 fb 11 6d 90       	vmovsd %xmm5,-0x70(%rbp)
    4d5f:	c5 d3 5e c9          	vdivsd %xmm1,%xmm5,%xmm1
    4d63:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
  { return __builtin_cosf(__x); }
    4d67:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    4d6b:	c5 fa 11 4d a0       	vmovss %xmm1,-0x60(%rbp)
    4d70:	e8 8b d3 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    4d75:	c5 fa 10 4d a0       	vmovss -0x60(%rbp),%xmm1
    4d7a:	c5 fa 11 45 b0       	vmovss %xmm0,-0x50(%rbp)
    4d7f:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    4d84:	e8 97 d3 ff ff       	call   2120 <sinf@plt>
        for (int p = 0; p < m; p++)
    4d89:	41 83 fd 7f          	cmp    $0x7f,%r13d
    4d8d:	44 8b 4d 88          	mov    -0x78(%rbp),%r9d
    4d91:	8b 45 80             	mov    -0x80(%rbp),%eax
    4d94:	c5 fb 10 6d 90       	vmovsd -0x70(%rbp),%xmm5
    4d99:	44 8b 5d b8          	mov    -0x48(%rbp),%r11d
    4d9d:	c5 78 28 d8          	vmovaps %xmm0,%xmm11
    4da1:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    4da6:	0f 8e a2 fa ff ff    	jle    484e <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xe0e>
    4dac:	c5 fa 10 15 58 12 00 	vmovss 0x1258(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    4db3:	00 
    4db4:	c5 7a 10 55 b0       	vmovss -0x50(%rbp),%xmm10
    4db9:	48 63 55 98          	movslq -0x68(%rbp),%rdx
    4dbd:	4c 89 75 88          	mov    %r14,-0x78(%rbp)
    4dc1:	48 89 de             	mov    %rbx,%rsi
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    4dc4:	c5 f8 28 e2          	vmovaps %xmm2,%xmm4
    4dc8:	4c 89 7d 98          	mov    %r15,-0x68(%rbp)
    4dcc:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    4dd0:	48 89 d1             	mov    %rdx,%rcx
    4dd3:	48 c1 e2 09          	shl    $0x9,%rdx
    4dd7:	44 89 4d 80          	mov    %r9d,-0x80(%rbp)
    4ddb:	45 31 c9             	xor    %r9d,%r9d
    4dde:	49 89 d2             	mov    %rdx,%r10
    4de1:	41 89 c8             	mov    %ecx,%r8d
    4de4:	48 8d 3c 13          	lea    (%rbx,%rdx,1),%rdi
    4de8:	4c 89 6d 90          	mov    %r13,-0x70(%rbp)
    4dec:	4c 89 55 b0          	mov    %r10,-0x50(%rbp)
    4df0:	41 c1 e0 06          	shl    $0x6,%r8d
    4df4:	49 83 c2 20          	add    $0x20,%r10
    4df8:	4c 8b 75 b0          	mov    -0x50(%rbp),%r14
    4dfc:	4c 89 55 a0          	mov    %r10,-0x60(%rbp)
    4e00:	4c 89 fa             	mov    %r15,%rdx
    4e03:	45 31 d2             	xor    %r10d,%r10d
    4e06:	4c 8b 7d a0          	mov    -0x60(%rbp),%r15
    4e0a:	44 89 65 b0          	mov    %r12d,-0x50(%rbp)
    4e0e:	45 89 c4             	mov    %r8d,%r12d
    4e11:	44 89 9d 7c ff ff ff 	mov    %r11d,-0x84(%rbp)
    4e18:	41 89 cb             	mov    %ecx,%r11d
    4e1b:	89 45 b8             	mov    %eax,-0x48(%rbp)
    4e1e:	eb 04                	jmp    4e24 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x13e4>
    4e20:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
                for (int q = 0; q < s; q += 2)
    4e24:	48 8d 4a 10          	lea    0x10(%rdx),%rcx
    4e28:	48 89 f0             	mov    %rsi,%rax
    4e2b:	48 29 c8             	sub    %rcx,%rax
    4e2e:	48 83 c0 0c          	add    $0xc,%rax
    4e32:	48 3d 18 02 00 00    	cmp    $0x218,%rax
    4e38:	48 89 f8             	mov    %rdi,%rax
    4e3b:	41 0f 97 c0          	seta   %r8b
    4e3f:	48 29 c8             	sub    %rcx,%rax
    4e42:	48 83 c0 0c          	add    $0xc,%rax
    4e46:	48 3d 18 02 00 00    	cmp    $0x218,%rax
    4e4c:	0f 97 c0             	seta   %al
    4e4f:	41 84 c0             	test   %al,%r8b
    4e52:	0f 84 b8 04 00 00    	je     5310 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x18d0>
    4e58:	4b 8d 04 0f          	lea    (%r15,%r9,1),%rax
    4e5c:	4b 8d 0c 0e          	lea    (%r14,%r9,1),%rcx
    4e60:	4c 39 c8             	cmp    %r9,%rax
    4e63:	4d 8d 41 20          	lea    0x20(%r9),%r8
    4e67:	0f 9e c0             	setle  %al
    4e6a:	49 39 c8             	cmp    %rcx,%r8
    4e6d:	0f 9e c1             	setle  %cl
    4e70:	08 c1                	or     %al,%cl
    4e72:	0f 84 98 04 00 00    	je     5310 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x18d0>
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4e78:	c5 fc 10 b2 00 02 00 	vmovups 0x200(%rdx),%ymm6
    4e7f:	00 
    4e80:	c5 e0 14 cc          	vunpcklps %xmm4,%xmm3,%xmm1
    4e84:	c5 d8 14 c3          	vunpcklps %xmm3,%xmm4,%xmm0
                    const complex_t a = y[q + s * (2 * p + 0)];
    4e88:	c5 fc 10 3a          	vmovups (%rdx),%ymm7
    4e8c:	c5 f0 16 c9          	vmovlhps %xmm1,%xmm1,%xmm1
    4e90:	c5 f8 16 c0          	vmovlhps %xmm0,%xmm0,%xmm0
    4e94:	c4 e3 75 18 c9 01    	vinsertf128 $0x1,%xmm1,%ymm1,%ymm1
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4e9a:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4ea0:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4ea6:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4eaa:	c4 e3 7d 18 c0 01    	vinsertf128 $0x1,%xmm0,%ymm0,%ymm0
    4eb0:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4eb5:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4eb9:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4ebd:	c5 7c 11 0e          	vmovups %ymm9,(%rsi)
                    x[q + s * (p + m)] = a - b;
    4ec1:	c5 fc 11 3f          	vmovups %ymm7,(%rdi)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4ec5:	c5 fc 10 b2 20 02 00 	vmovups 0x220(%rdx),%ymm6
    4ecc:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4ecd:	c5 fc 10 7a 20       	vmovups 0x20(%rdx),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4ed2:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4ed8:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4ede:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4ee2:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4ee7:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4eeb:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4eef:	c5 7c 11 4e 20       	vmovups %ymm9,0x20(%rsi)
                    x[q + s * (p + m)] = a - b;
    4ef4:	c5 fc 11 7f 20       	vmovups %ymm7,0x20(%rdi)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4ef9:	c5 fc 10 b2 40 02 00 	vmovups 0x240(%rdx),%ymm6
    4f00:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4f01:	c5 fc 10 7a 40       	vmovups 0x40(%rdx),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4f06:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4f0c:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4f12:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4f16:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4f1b:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4f1f:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4f23:	c5 7c 11 4e 40       	vmovups %ymm9,0x40(%rsi)
                    x[q + s * (p + m)] = a - b;
    4f28:	c5 fc 11 7f 40       	vmovups %ymm7,0x40(%rdi)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4f2d:	c5 fc 10 b2 60 02 00 	vmovups 0x260(%rdx),%ymm6
    4f34:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4f35:	c5 fc 10 7a 60       	vmovups 0x60(%rdx),%ymm7
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4f3a:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4f40:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4f46:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4f4a:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4f4f:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4f53:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4f57:	c5 7c 11 4e 60       	vmovups %ymm9,0x60(%rsi)
                    x[q + s * (p + m)] = a - b;
    4f5c:	c5 fc 11 7f 60       	vmovups %ymm7,0x60(%rdi)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4f61:	c5 fc 10 b2 80 02 00 	vmovups 0x280(%rdx),%ymm6
    4f68:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4f69:	c5 fc 10 ba 80 00 00 	vmovups 0x80(%rdx),%ymm7
    4f70:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4f71:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4f77:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4f7d:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4f81:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4f86:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4f8a:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4f8e:	c5 7c 11 8e 80 00 00 	vmovups %ymm9,0x80(%rsi)
    4f95:	00 
                    x[q + s * (p + m)] = a - b;
    4f96:	c5 fc 11 bf 80 00 00 	vmovups %ymm7,0x80(%rdi)
    4f9d:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4f9e:	c5 fc 10 b2 a0 02 00 	vmovups 0x2a0(%rdx),%ymm6
    4fa5:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4fa6:	c5 fc 10 ba a0 00 00 	vmovups 0xa0(%rdx),%ymm7
    4fad:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4fae:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4fb4:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4fba:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4fbe:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    4fc3:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    4fc7:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    4fcb:	c5 7c 11 8e a0 00 00 	vmovups %ymm9,0xa0(%rsi)
    4fd2:	00 
                    x[q + s * (p + m)] = a - b;
    4fd3:	c5 fc 11 bf a0 00 00 	vmovups %ymm7,0xa0(%rdi)
    4fda:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4fdb:	c5 fc 10 b2 c0 02 00 	vmovups 0x2c0(%rdx),%ymm6
    4fe2:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    4fe3:	c5 fc 10 ba c0 00 00 	vmovups 0xc0(%rdx),%ymm7
    4fea:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    4feb:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    4ff1:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    4ff7:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    4ffb:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    5000:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5004:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    5008:	c5 7c 11 8e c0 00 00 	vmovups %ymm9,0xc0(%rsi)
    500f:	00 
                    x[q + s * (p + m)] = a - b;
    5010:	c5 fc 11 bf c0 00 00 	vmovups %ymm7,0xc0(%rdi)
    5017:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5018:	c5 fc 10 b2 e0 02 00 	vmovups 0x2e0(%rdx),%ymm6
    501f:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    5020:	c5 fc 10 ba e0 00 00 	vmovups 0xe0(%rdx),%ymm7
    5027:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5028:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    502e:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    5034:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    5038:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    503d:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5041:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    5045:	c5 7c 11 8e e0 00 00 	vmovups %ymm9,0xe0(%rsi)
    504c:	00 
                    x[q + s * (p + m)] = a - b;
    504d:	c5 fc 11 bf e0 00 00 	vmovups %ymm7,0xe0(%rdi)
    5054:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    5055:	c5 fc 10 ba 00 01 00 	vmovups 0x100(%rdx),%ymm7
    505c:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    505d:	c5 fc 10 b2 00 03 00 	vmovups 0x300(%rdx),%ymm6
    5064:	00 
    5065:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    506b:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    5071:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    5075:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    507a:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    507e:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    5082:	c5 7c 11 8e 00 01 00 	vmovups %ymm9,0x100(%rsi)
    5089:	00 
                    x[q + s * (p + m)] = a - b;
    508a:	c5 fc 11 bf 00 01 00 	vmovups %ymm7,0x100(%rdi)
    5091:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5092:	c5 fc 10 b2 20 03 00 	vmovups 0x320(%rdx),%ymm6
    5099:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    509a:	c5 fc 10 ba 20 01 00 	vmovups 0x120(%rdx),%ymm7
    50a1:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    50a2:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    50a8:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    50ae:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    50b2:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    50b7:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    50bb:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    50bf:	c5 7c 11 8e 20 01 00 	vmovups %ymm9,0x120(%rsi)
    50c6:	00 
                    x[q + s * (p + m)] = a - b;
    50c7:	c5 fc 11 bf 20 01 00 	vmovups %ymm7,0x120(%rdi)
    50ce:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    50cf:	c5 fc 10 b2 40 03 00 	vmovups 0x340(%rdx),%ymm6
    50d6:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    50d7:	c5 fc 10 ba 40 01 00 	vmovups 0x140(%rdx),%ymm7
    50de:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    50df:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    50e5:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    50eb:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    50ef:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    50f4:	c5 44 58 ce          	vaddps %ymm6,%ymm7,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    50f8:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    50fc:	c5 7c 11 8e 40 01 00 	vmovups %ymm9,0x140(%rsi)
    5103:	00 
                    x[q + s * (p + m)] = a - b;
    5104:	c5 fc 11 bf 40 01 00 	vmovups %ymm7,0x140(%rdi)
    510b:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    510c:	c5 fc 10 b2 60 03 00 	vmovups 0x360(%rdx),%ymm6
    5113:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    5114:	c5 fc 10 ba 60 01 00 	vmovups 0x160(%rdx),%ymm7
    511b:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    511c:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    5122:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    5128:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    512c:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    5131:	c5 4c 58 cf          	vaddps %ymm7,%ymm6,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5135:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    5139:	c5 7c 11 8e 60 01 00 	vmovups %ymm9,0x160(%rsi)
    5140:	00 
                    x[q + s * (p + m)] = a - b;
    5141:	c5 fc 11 bf 60 01 00 	vmovups %ymm7,0x160(%rdi)
    5148:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5149:	c5 fc 10 b2 80 03 00 	vmovups 0x380(%rdx),%ymm6
    5150:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    5151:	c5 fc 10 ba 80 01 00 	vmovups 0x180(%rdx),%ymm7
    5158:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5159:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    515f:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    5165:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    5169:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    516e:	c5 4c 58 cf          	vaddps %ymm7,%ymm6,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5172:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    5176:	c5 7c 11 8e 80 01 00 	vmovups %ymm9,0x180(%rsi)
    517d:	00 
                    x[q + s * (p + m)] = a - b;
    517e:	c5 fc 11 bf 80 01 00 	vmovups %ymm7,0x180(%rdi)
    5185:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5186:	c5 fc 10 b2 a0 03 00 	vmovups 0x3a0(%rdx),%ymm6
    518d:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    518e:	c5 fc 10 ba a0 01 00 	vmovups 0x1a0(%rdx),%ymm7
    5195:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5196:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    519c:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    51a2:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    51a6:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    51ab:	c5 4c 58 cf          	vaddps %ymm7,%ymm6,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    51af:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    51b3:	c5 7c 11 8e a0 01 00 	vmovups %ymm9,0x1a0(%rsi)
    51ba:	00 
                    x[q + s * (p + m)] = a - b;
    51bb:	c5 fc 11 bf a0 01 00 	vmovups %ymm7,0x1a0(%rdi)
    51c2:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    51c3:	c5 fc 10 b2 c0 03 00 	vmovups 0x3c0(%rdx),%ymm6
    51ca:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    51cb:	c5 fc 10 ba c0 01 00 	vmovups 0x1c0(%rdx),%ymm7
    51d2:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    51d3:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    51d9:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    51df:	c5 cc 59 f1          	vmulps %ymm1,%ymm6,%ymm6
    51e3:	c4 c2 7d b6 f1       	vfmaddsub231ps %ymm9,%ymm0,%ymm6
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    51e8:	c5 4c 58 cf          	vaddps %ymm7,%ymm6,%ymm9
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    51ec:	c5 c4 5c fe          	vsubps %ymm6,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    51f0:	c5 7c 11 8e c0 01 00 	vmovups %ymm9,0x1c0(%rsi)
    51f7:	00 
                    x[q + s * (p + m)] = a - b;
    51f8:	c5 fc 11 bf c0 01 00 	vmovups %ymm7,0x1c0(%rdi)
    51ff:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5200:	c5 fc 10 b2 e0 03 00 	vmovups 0x3e0(%rdx),%ymm6
    5207:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    5208:	c5 fc 10 ba e0 01 00 	vmovups 0x1e0(%rdx),%ymm7
    520f:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5210:	c4 63 7d 04 ce a0    	vpermilps $0xa0,%ymm6,%ymm9
    5216:	c4 e3 7d 04 f6 f5    	vpermilps $0xf5,%ymm6,%ymm6
    521c:	c5 cc 59 c9          	vmulps %ymm1,%ymm6,%ymm1
    5220:	c4 c2 75 96 c1       	vfmaddsub132ps %ymm9,%ymm1,%ymm0
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    5225:	c5 fc 58 cf          	vaddps %ymm7,%ymm0,%ymm1
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5229:	c5 c4 5c f8          	vsubps %ymm0,%ymm7,%ymm7
                    x[q + s * (p + 0)] = a + b;
    522d:	c5 fc 11 8e e0 01 00 	vmovups %ymm1,0x1e0(%rsi)
    5234:	00 
                    x[q + s * (p + m)] = a - b;
    5235:	c5 fc 11 bf e0 01 00 	vmovups %ymm7,0x1e0(%rdi)
    523c:	00 
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    523d:	c4 c1 62 59 c2       	vmulss %xmm10,%xmm3,%xmm0
        for (int p = 0; p < m; p++)
    5242:	41 ff c2             	inc    %r10d
    5245:	49 81 c1 00 02 00 00 	add    $0x200,%r9
    524c:	48 81 c7 00 02 00 00 	add    $0x200,%rdi
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5253:	c4 c1 62 59 db       	vmulss %xmm11,%xmm3,%xmm3
        for (int p = 0; p < m; p++)
    5258:	48 81 c2 00 04 00 00 	add    $0x400,%rdx
    525f:	48 81 c6 00 02 00 00 	add    $0x200,%rsi
    5266:	41 83 c4 40          	add    $0x40,%r12d
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    526a:	c4 c2 59 b9 c3       	vfmadd231ss %xmm11,%xmm4,%xmm0
    526f:	c4 c2 61 9b e2       	vfmsub132ss %xmm10,%xmm3,%xmm4
        for (int p = 0; p < m; p++)
    5274:	45 39 d3             	cmp    %r10d,%r11d
    5277:	0f 8f a3 fb ff ff    	jg     4e20 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x13e0>
    const float angle = 2 * M_PI / n;
    527d:	c5 bb 2a 4d bc       	vcvtsi2sdl -0x44(%rbp),%xmm8,%xmm1
    5282:	44 8b 4d 80          	mov    -0x80(%rbp),%r9d
    5286:	8b 45 b8             	mov    -0x48(%rbp),%eax
    5289:	44 8b 9d 7c ff ff ff 	mov    -0x84(%rbp),%r11d
    5290:	4c 8b 75 88          	mov    -0x78(%rbp),%r14
    5294:	4c 8b 7d 98          	mov    -0x68(%rbp),%r15
    5298:	4c 8b 6d 90          	mov    -0x70(%rbp),%r13
    529c:	44 89 8d 78 ff ff ff 	mov    %r9d,-0x88(%rbp)
    52a3:	89 45 80             	mov    %eax,-0x80(%rbp)
    52a6:	44 8b 65 b0          	mov    -0x50(%rbp),%r12d
    52aa:	c5 d3 5e c9          	vdivsd %xmm1,%xmm5,%xmm1
    52ae:	44 89 5d 88          	mov    %r11d,-0x78(%rbp)
    52b2:	c5 fa 11 55 90       	vmovss %xmm2,-0x70(%rbp)
    52b7:	c5 fb 11 6d 98       	vmovsd %xmm5,-0x68(%rbp)
    52bc:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
  { return __builtin_cosf(__x); }
    52c0:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    52c4:	c5 fa 11 4d a0       	vmovss %xmm1,-0x60(%rbp)
    52c9:	c5 f8 77             	vzeroupper
    52cc:	e8 2f ce ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    52d1:	c5 fa 10 4d a0       	vmovss -0x60(%rbp),%xmm1
    52d6:	c5 fa 11 45 b0       	vmovss %xmm0,-0x50(%rbp)
    52db:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    52e0:	e8 3b ce ff ff       	call   2120 <sinf@plt>
    52e5:	44 8b 5d 88          	mov    -0x78(%rbp),%r11d
    52e9:	8b 45 80             	mov    -0x80(%rbp),%eax
    52ec:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    52f1:	c5 7a 10 55 b0       	vmovss -0x50(%rbp),%xmm10
    52f6:	c5 fb 10 6d 98       	vmovsd -0x68(%rbp),%xmm5
    52fb:	c5 78 28 d8          	vmovaps %xmm0,%xmm11
        for (int p = 0; p < m; p++)
    52ff:	c5 fa 10 55 90       	vmovss -0x70(%rbp),%xmm2
    5304:	44 8b 8d 78 ff ff ff 	mov    -0x88(%rbp),%r9d
    530b:	e9 ad f5 ff ff       	jmp    48bd <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0xe7d>
    5310:	49 63 c4             	movslq %r12d,%rax
                    x[q + s * (p + m)] = a - b;
    5313:	48 89 f1             	mov    %rsi,%rcx
    5316:	4c 8d aa 00 02 00 00 	lea    0x200(%rdx),%r13
    531d:	4c 8d 04 c3          	lea    (%rbx,%rax,8),%r8
    5321:	48 89 d0             	mov    %rdx,%rax
    5324:	0f 1f 40 00          	nopl   0x0(%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5328:	c5 fa 10 80 04 02 00 	vmovss 0x204(%rax),%xmm0
    532f:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    5330:	c5 7a 10 30          	vmovss (%rax),%xmm14
                for (int q = 0; q < s; q += 2)
    5334:	48 83 c0 10          	add    $0x10,%rax
    5338:	48 83 c1 10          	add    $0x10,%rcx
                    const complex_t a = y[q + s * (2 * p + 0)];
    533c:	c5 7a 10 60 f4       	vmovss -0xc(%rax),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    5341:	c5 7a 10 48 f8       	vmovss -0x8(%rax),%xmm9
                for (int q = 0; q < s; q += 2)
    5346:	49 83 c0 10          	add    $0x10,%r8
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    534a:	c5 5a 59 e8          	vmulss %xmm0,%xmm4,%xmm13
    534e:	c5 fa 10 88 f0 01 00 	vmovss 0x1f0(%rax),%xmm1
    5355:	00 
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    5356:	c5 fa 10 70 fc       	vmovss -0x4(%rax),%xmm6
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    535b:	c5 e2 59 c0          	vmulss %xmm0,%xmm3,%xmm0
    535f:	c5 7a 10 b8 fc 01 00 	vmovss 0x1fc(%rax),%xmm15
    5366:	00 
    5367:	c4 c1 5a 59 ff       	vmulss %xmm15,%xmm4,%xmm7
    536c:	c4 41 62 59 ff       	vmulss %xmm15,%xmm3,%xmm15
    5371:	c4 62 61 b9 e9       	vfmadd231ss %xmm1,%xmm3,%xmm13
    5376:	c4 e2 79 9b cc       	vfmsub132ss %xmm4,%xmm0,%xmm1
    537b:	c5 fa 10 80 f8 01 00 	vmovss 0x1f8(%rax),%xmm0
    5382:	00 
    5383:	c4 e2 61 b9 f8       	vfmadd231ss %xmm0,%xmm3,%xmm7
    5388:	c4 e2 01 9b c4       	vfmsub132ss %xmm4,%xmm15,%xmm0
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    538d:	c4 41 72 58 fe       	vaddss %xmm14,%xmm1,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5392:	c5 8a 5c c9          	vsubss %xmm1,%xmm14,%xmm1
                    x[q + s * (p + 0)] = a + b;
    5396:	c5 7a 11 79 f0       	vmovss %xmm15,-0x10(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    539b:	c4 41 12 58 fc       	vaddss %xmm12,%xmm13,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    53a0:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    53a5:	c5 7a 11 79 f4       	vmovss %xmm15,-0xc(%rcx)
                    x[q + s * (p + m)] = a - b;
    53aa:	c4 c1 7a 11 48 f0    	vmovss %xmm1,-0x10(%r8)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    53b0:	c4 c1 7a 58 c9       	vaddss %xmm9,%xmm0,%xmm1
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    53b5:	c5 b2 5c c0          	vsubss %xmm0,%xmm9,%xmm0
                    x[q + s * (p + m)] = a - b;
    53b9:	c4 41 7a 11 60 f4    	vmovss %xmm12,-0xc(%r8)
                    x[(q + 1) + s * (p + 0)] = c + d;
    53bf:	c5 fa 11 49 f8       	vmovss %xmm1,-0x8(%rcx)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    53c4:	c5 c2 58 ce          	vaddss %xmm6,%xmm7,%xmm1
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    53c8:	c5 ca 5c f7          	vsubss %xmm7,%xmm6,%xmm6
                    x[(q + 1) + s * (p + 0)] = c + d;
    53cc:	c5 fa 11 49 fc       	vmovss %xmm1,-0x4(%rcx)
                    x[(q + 1) + s * (p + m)] = c - d;
    53d1:	c4 c1 7a 11 40 f8    	vmovss %xmm0,-0x8(%r8)
    53d7:	c4 c1 7a 11 70 fc    	vmovss %xmm6,-0x4(%r8)
                for (int q = 0; q < s; q += 2)
    53dd:	4c 39 e8             	cmp    %r13,%rax
    53e0:	0f 85 42 ff ff ff    	jne    5328 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x18e8>
    53e6:	e9 52 fe ff ff       	jmp    523d <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x17fd>
    const float angle = 2 * M_PI / n;
    53eb:	c5 bb 2a 45 98       	vcvtsi2sdl -0x68(%rbp),%xmm8,%xmm0
    const int m = n / 2;
    53f0:	85 ff                	test   %edi,%edi
    53f2:	8d 97 ff 00 00 00    	lea    0xff(%rdi),%edx
        fftr(n / 2, 2 * s, !eo, y, x);
    53f8:	4d 89 f8             	mov    %r15,%r8
    const int m = n / 2;
    53fb:	0f 49 d7             	cmovns %edi,%edx
        fftr(n / 2, 2 * s, !eo, y, x);
    53fe:	48 89 d9             	mov    %rbx,%rcx
    5401:	be 00 01 00 00       	mov    $0x100,%esi
    5406:	44 89 5d b8          	mov    %r11d,-0x48(%rbp)
    const float angle = 2 * M_PI / n;
    540a:	c5 fb 10 2d ce 0c 00 	vmovsd 0xcce(%rip),%xmm5        # 60e0 <_IO_stdin_used+0xe0>
    5411:	00 
    5412:	89 45 80             	mov    %eax,-0x80(%rbp)
    const int m = n / 2;
    5415:	89 d7                	mov    %edx,%edi
        fftr(n / 2, 2 * s, !eo, y, x);
    5417:	31 d2                	xor    %edx,%edx
    5419:	44 89 4d 88          	mov    %r9d,-0x78(%rbp)
    const float angle = 2 * M_PI / n;
    541d:	c5 d3 5e c0          	vdivsd %xmm0,%xmm5,%xmm0
    const int m = n / 2;
    5421:	c1 ff 08             	sar    $0x8,%edi
    const float angle = 2 * M_PI / n;
    5424:	c5 fb 11 6d 90       	vmovsd %xmm5,-0x70(%rbp)
        fftr(n / 2, 2 * s, !eo, y, x);
    5429:	89 7d b0             	mov    %edi,-0x50(%rbp)
    const float angle = 2 * M_PI / n;
    542c:	c5 fb 5a d0          	vcvtsd2ss %xmm0,%xmm0,%xmm2
    5430:	c5 fa 11 55 a0       	vmovss %xmm2,-0x60(%rbp)
        fftr(n / 2, 2 * s, !eo, y, x);
    5435:	e8 56 e1 ff ff       	call   3590 <_Z4fftriibP9complex_tS0_>
  { return __builtin_cosf(__x); }
    543a:	c5 fa 10 45 a0       	vmovss -0x60(%rbp),%xmm0
    543f:	e8 bc cc ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    5444:	c5 fa 10 6d a0       	vmovss -0x60(%rbp),%xmm5
    5449:	c5 fa 10 15 9f 0c 00 	vmovss 0xc9f(%rip),%xmm2        # 60f0 <_IO_stdin_used+0xf0>
    5450:	00 
    5451:	c5 fa 11 85 7c ff ff 	vmovss %xmm0,-0x84(%rbp)
    5458:	ff 
    5459:	c5 d0 57 c2          	vxorps %xmm2,%xmm5,%xmm0
    545d:	c5 f8 29 55 c0       	vmovaps %xmm2,-0x40(%rbp)
  { return __builtin_sinf(__x); }
    5462:	e8 b9 cc ff ff       	call   2120 <sinf@plt>
        for (int p = 0; p < m; p++)
    5467:	44 8b 4d 88          	mov    -0x78(%rbp),%r9d
    546b:	8b 45 80             	mov    -0x80(%rbp),%eax
    546e:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    5473:	41 81 fd ff 00 00 00 	cmp    $0xff,%r13d
    547a:	c5 fb 10 6d 90       	vmovsd -0x70(%rbp),%xmm5
    547f:	44 8b 5d b8          	mov    -0x48(%rbp),%r11d
    5483:	c5 78 28 c8          	vmovaps %xmm0,%xmm9
    5487:	0f 8e bd f8 ff ff    	jle    4d4a <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x130a>
    548d:	48 63 55 b0          	movslq -0x50(%rbp),%rdx
    5491:	4c 89 75 80          	mov    %r14,-0x80(%rbp)
    5495:	45 31 d2             	xor    %r10d,%r10d
    5498:	48 89 de             	mov    %rbx,%rsi
    complex_t(const float &x, const float &y) : Re(x), Im(y) {}
    549b:	c5 fa 10 15 69 0b 00 	vmovss 0xb69(%rip),%xmm2        # 600c <_IO_stdin_used+0xc>
    54a2:	00 
    54a3:	48 89 5d 90          	mov    %rbx,-0x70(%rbp)
    54a7:	4c 89 f9             	mov    %r15,%rcx
    54aa:	44 89 d3             	mov    %r10d,%ebx
    54ad:	89 d7                	mov    %edx,%edi
    54af:	48 c1 e2 0a          	shl    $0xa,%rdx
    54b3:	4c 89 6d 88          	mov    %r13,-0x78(%rbp)
    54b7:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    54bb:	4c 8d 42 20          	lea    0x20(%rdx),%r8
    54bf:	c1 e7 07             	shl    $0x7,%edi
    54c2:	44 89 65 b8          	mov    %r12d,-0x48(%rbp)
    54c6:	c5 f8 28 da          	vmovaps %xmm2,%xmm3
    54ca:	4c 89 45 a0          	mov    %r8,-0x60(%rbp)
    54ce:	45 31 d2             	xor    %r10d,%r10d
    54d1:	49 89 d5             	mov    %rdx,%r13
    54d4:	41 89 fc             	mov    %edi,%r12d
    54d7:	c5 fa 10 bd 7c ff ff 	vmovss -0x84(%rbp),%xmm7
    54de:	ff 
                for (int q = 0; q < s; q += 2)
    54df:	4c 8d 46 10          	lea    0x10(%rsi),%r8
    54e3:	48 89 cf             	mov    %rcx,%rdi
    54e6:	49 8d 54 0d 00       	lea    0x0(%r13,%rcx,1),%rdx
    54eb:	4c 29 c7             	sub    %r8,%rdi
    54ee:	48 83 c7 0c          	add    $0xc,%rdi
    54f2:	48 81 ff 18 04 00 00 	cmp    $0x418,%rdi
    54f9:	48 89 d7             	mov    %rdx,%rdi
    54fc:	41 0f 97 c6          	seta   %r14b
    5500:	4c 29 c7             	sub    %r8,%rdi
    5503:	48 83 c7 0c          	add    $0xc,%rdi
    5507:	48 81 ff 18 04 00 00 	cmp    $0x418,%rdi
    550e:	40 0f 97 c7          	seta   %dil
    5512:	41 84 fe             	test   %dil,%r14b
    5515:	0f 84 e5 00 00 00    	je     5600 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1bc0>
    551b:	48 8b 7d a0          	mov    -0x60(%rbp),%rdi
    551f:	4f 8d 44 15 00       	lea    0x0(%r13,%r10,1),%r8
    5524:	4d 8d 72 20          	lea    0x20(%r10),%r14
    5528:	4c 01 d7             	add    %r10,%rdi
    552b:	4c 39 d7             	cmp    %r10,%rdi
    552e:	40 0f 9e c7          	setle  %dil
    5532:	4d 39 c6             	cmp    %r8,%r14
    5535:	41 0f 9e c0          	setle  %r8b
    5539:	41 08 f8             	or     %dil,%r8b
    553c:	0f 84 be 00 00 00    	je     5600 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1bc0>
    5542:	c5 e0 14 f0          	vunpcklps %xmm0,%xmm3,%xmm6
    5546:	c5 f8 14 e3          	vunpcklps %xmm3,%xmm0,%xmm4
    554a:	4c 8d 86 00 04 00 00 	lea    0x400(%rsi),%r8
                    x[q + s * (p + m)] = a - b;
    5551:	31 ff                	xor    %edi,%edi
    5553:	c5 c8 16 f6          	vmovlhps %xmm6,%xmm6,%xmm6
    5557:	c5 d8 16 e4          	vmovlhps %xmm4,%xmm4,%xmm4
    555b:	c4 e3 4d 18 f6 01    	vinsertf128 $0x1,%xmm6,%ymm6,%ymm6
    5561:	c4 e3 5d 18 e4 01    	vinsertf128 $0x1,%xmm4,%ymm4,%ymm4
    5567:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    556e:	00 00 00 00 
    5572:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    5579:	00 00 00 00 
    557d:	0f 1f 00             	nopl   (%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5580:	c4 41 7c 10 14 38    	vmovups (%r8,%rdi,1),%ymm10
                    const complex_t a = y[q + s * (2 * p + 0)];
    5586:	c5 fc 10 0c 3e       	vmovups (%rsi,%rdi,1),%ymm1
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    558b:	c4 43 7d 04 da a0    	vpermilps $0xa0,%ymm10,%ymm11
    5591:	c4 43 7d 04 d2 f5    	vpermilps $0xf5,%ymm10,%ymm10
    5597:	c5 2c 59 d4          	vmulps %ymm4,%ymm10,%ymm10
    559b:	c4 62 2d 96 de       	vfmaddsub132ps %ymm6,%ymm10,%ymm11
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    55a0:	c5 24 58 d1          	vaddps %ymm1,%ymm11,%ymm10
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    55a4:	c4 c1 74 5c cb       	vsubps %ymm11,%ymm1,%ymm1
                    x[q + s * (p + 0)] = a + b;
    55a9:	c5 7c 11 14 39       	vmovups %ymm10,(%rcx,%rdi,1)
                    x[q + s * (p + m)] = a - b;
    55ae:	c5 fc 11 0c 3a       	vmovups %ymm1,(%rdx,%rdi,1)
                for (int q = 0; q < s; q += 2)
    55b3:	48 83 c7 20          	add    $0x20,%rdi
    55b7:	48 81 ff 00 04 00 00 	cmp    $0x400,%rdi
    55be:	75 c0                	jne    5580 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1b40>
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    55c0:	c5 fa 59 cf          	vmulss %xmm7,%xmm0,%xmm1
        for (int p = 0; p < m; p++)
    55c4:	ff c3                	inc    %ebx
    55c6:	49 81 c2 00 04 00 00 	add    $0x400,%r10
    55cd:	48 81 c6 00 08 00 00 	add    $0x800,%rsi
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    55d4:	c4 c1 7a 59 c1       	vmulss %xmm9,%xmm0,%xmm0
        for (int p = 0; p < m; p++)
    55d9:	48 81 c1 00 04 00 00 	add    $0x400,%rcx
    55e0:	41 83 ec 80          	sub    $0xffffff80,%r12d
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    55e4:	c4 c2 61 b9 c9       	vfmadd231ss %xmm9,%xmm3,%xmm1
    55e9:	c4 e2 79 9b df       	vfmsub132ss %xmm7,%xmm0,%xmm3
        for (int p = 0; p < m; p++)
    55ee:	39 5d b0             	cmp    %ebx,-0x50(%rbp)
    55f1:	0f 8e e5 00 00 00    	jle    56dc <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1c9c>
    55f7:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    55fb:	e9 df fe ff ff       	jmp    54df <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1a9f>
    5600:	49 63 d4             	movslq %r12d,%rdx
                    x[q + s * (p + m)] = a - b;
    5603:	48 89 cf             	mov    %rcx,%rdi
    5606:	4c 8d b6 00 04 00 00 	lea    0x400(%rsi),%r14
    560d:	4d 8d 04 d7          	lea    (%r15,%rdx,8),%r8
    5611:	48 89 f2             	mov    %rsi,%rdx
    5614:	0f 1f 40 00          	nopl   0x0(%rax)
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    5618:	c5 fa 10 8a 04 04 00 	vmovss 0x404(%rdx),%xmm1
    561f:	00 
                    const complex_t a = y[q + s * (2 * p + 0)];
    5620:	c5 7a 10 32          	vmovss (%rdx),%xmm14
                for (int q = 0; q < s; q += 2)
    5624:	48 83 c2 10          	add    $0x10,%rdx
    5628:	48 83 c7 10          	add    $0x10,%rdi
                    const complex_t a = y[q + s * (2 * p + 0)];
    562c:	c5 7a 10 62 f4       	vmovss -0xc(%rdx),%xmm12
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    5631:	c5 7a 10 5a f8       	vmovss -0x8(%rdx),%xmm11
                for (int q = 0; q < s; q += 2)
    5636:	49 83 c0 10          	add    $0x10,%r8
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    563a:	c5 62 59 e9          	vmulss %xmm1,%xmm3,%xmm13
    563e:	c5 fa 10 a2 f0 03 00 	vmovss 0x3f0(%rdx),%xmm4
    5645:	00 
                    const complex_t c = y[(q + 1) + s * (2 * p + 0)];
    5646:	c5 fa 10 72 fc       	vmovss -0x4(%rdx),%xmm6
    return complex_t(x.Re * y.Re - x.Im * y.Im, x.Re * y.Im + x.Im * y.Re);
    564b:	c5 fa 59 c9          	vmulss %xmm1,%xmm0,%xmm1
    564f:	c5 7a 10 ba fc 03 00 	vmovss 0x3fc(%rdx),%xmm15
    5656:	00 
    5657:	c4 41 62 59 d7       	vmulss %xmm15,%xmm3,%xmm10
    565c:	c4 41 7a 59 ff       	vmulss %xmm15,%xmm0,%xmm15
    5661:	c4 62 79 b9 ec       	vfmadd231ss %xmm4,%xmm0,%xmm13
    5666:	c4 e2 71 9b e3       	vfmsub132ss %xmm3,%xmm1,%xmm4
    566b:	c5 fa 10 8a f8 03 00 	vmovss 0x3f8(%rdx),%xmm1
    5672:	00 
    5673:	c4 62 79 b9 d1       	vfmadd231ss %xmm1,%xmm0,%xmm10
    5678:	c4 e2 01 9b cb       	vfmsub132ss %xmm3,%xmm15,%xmm1
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    567d:	c4 41 5a 58 fe       	vaddss %xmm14,%xmm4,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5682:	c5 8a 5c e4          	vsubss %xmm4,%xmm14,%xmm4
                    x[q + s * (p + 0)] = a + b;
    5686:	c5 7a 11 7f f0       	vmovss %xmm15,-0x10(%rdi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    568b:	c4 41 12 58 fc       	vaddss %xmm12,%xmm13,%xmm15
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    5690:	c4 41 1a 5c e5       	vsubss %xmm13,%xmm12,%xmm12
                    x[q + s * (p + 0)] = a + b;
    5695:	c5 7a 11 7f f4       	vmovss %xmm15,-0xc(%rdi)
                    x[q + s * (p + m)] = a - b;
    569a:	c4 c1 7a 11 60 f0    	vmovss %xmm4,-0x10(%r8)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    56a0:	c4 c1 72 58 e3       	vaddss %xmm11,%xmm1,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    56a5:	c5 a2 5c c9          	vsubss %xmm1,%xmm11,%xmm1
                    x[q + s * (p + m)] = a - b;
    56a9:	c4 41 7a 11 60 f4    	vmovss %xmm12,-0xc(%r8)
                    x[(q + 1) + s * (p + 0)] = c + d;
    56af:	c5 fa 11 67 f8       	vmovss %xmm4,-0x8(%rdi)
    return complex_t(x.Re + y.Re, x.Im + y.Im);
    56b4:	c5 aa 58 e6          	vaddss %xmm6,%xmm10,%xmm4
    return complex_t(x.Re - y.Re, x.Im - y.Im);
    56b8:	c4 c1 4a 5c f2       	vsubss %xmm10,%xmm6,%xmm6
                    x[(q + 1) + s * (p + 0)] = c + d;
    56bd:	c5 fa 11 67 fc       	vmovss %xmm4,-0x4(%rdi)
                    x[(q + 1) + s * (p + m)] = c - d;
    56c2:	c4 c1 7a 11 48 f8    	vmovss %xmm1,-0x8(%r8)
    56c8:	c4 c1 7a 11 70 fc    	vmovss %xmm6,-0x4(%r8)
                for (int q = 0; q < s; q += 2)
    56ce:	49 39 d6             	cmp    %rdx,%r14
    56d1:	0f 85 41 ff ff ff    	jne    5618 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1bd8>
    56d7:	e9 e4 fe ff ff       	jmp    55c0 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1b80>
    const float angle = 2 * M_PI / n;
    56dc:	c5 bb 2a 4d a8       	vcvtsi2sdl -0x58(%rbp),%xmm8,%xmm1
    56e1:	44 89 8d 7c ff ff ff 	mov    %r9d,-0x84(%rbp)
    56e8:	4c 8b 75 80          	mov    -0x80(%rbp),%r14
    56ec:	44 8b 65 b8          	mov    -0x48(%rbp),%r12d
    56f0:	48 8b 5d 90          	mov    -0x70(%rbp),%rbx
    56f4:	89 45 b8             	mov    %eax,-0x48(%rbp)
    56f7:	44 89 5d 80          	mov    %r11d,-0x80(%rbp)
    56fb:	4c 8b 6d 88          	mov    -0x78(%rbp),%r13
    56ff:	c5 fb 11 6d 90       	vmovsd %xmm5,-0x70(%rbp)
    5704:	c5 d3 5e c9          	vdivsd %xmm1,%xmm5,%xmm1
    5708:	c5 fa 11 55 88       	vmovss %xmm2,-0x78(%rbp)
    570d:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
  { return __builtin_cosf(__x); }
    5711:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    5715:	c5 fa 11 4d a0       	vmovss %xmm1,-0x60(%rbp)
    571a:	c5 f8 77             	vzeroupper
    571d:	e8 de c9 ff ff       	call   2100 <cosf@plt>
        const complex_t wp_step{std::cos(angle), std::sin(-angle)};
    5722:	c5 fa 10 4d a0       	vmovss -0x60(%rbp),%xmm1
    5727:	c5 fa 11 45 b0       	vmovss %xmm0,-0x50(%rbp)
    572c:	c5 f0 57 45 c0       	vxorps -0x40(%rbp),%xmm1,%xmm0
  { return __builtin_sinf(__x); }
    5731:	e8 ea c9 ff ff       	call   2120 <sinf@plt>
    5736:	44 8b 5d 80          	mov    -0x80(%rbp),%r11d
    573a:	8b 45 b8             	mov    -0x48(%rbp),%eax
    573d:	c4 41 38 57 c0       	vxorps %xmm8,%xmm8,%xmm8
    5742:	c5 7a 10 55 b0       	vmovss -0x50(%rbp),%xmm10
    5747:	c5 fb 10 6d 90       	vmovsd -0x70(%rbp),%xmm5
    574c:	c5 78 28 d8          	vmovaps %xmm0,%xmm11
        for (int p = 0; p < m; p++)
    5750:	c5 fa 10 55 88       	vmovss -0x78(%rbp),%xmm2
    5755:	44 8b 8d 7c ff ff ff 	mov    -0x84(%rbp),%r9d
    575c:	e9 58 f6 ff ff       	jmp    4db9 <_Z22fft_stockham_recursivemRSt6vectorISt7complexIfESaIS1_EES4_+0x1379>
    5761:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    5768:	00 00 00 
    576b:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    5772:	00 00 00 
    5775:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    577c:	00 00 00 
    577f:	90                   	nop

0000000000005780 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>:
{
    5780:	55                   	push   %rbp
    5781:	48 89 e5             	mov    %rsp,%rbp
    5784:	41 57                	push   %r15
    5786:	41 56                	push   %r14
    5788:	41 55                	push   %r13
    578a:	41 54                	push   %r12
    578c:	49 89 f4             	mov    %rsi,%r12
    578f:	53                   	push   %rbx
    5790:	48 89 fb             	mov    %rdi,%rbx
    5793:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    5797:	48 81 ec 60 02 00 00 	sub    $0x260,%rsp
	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
    579e:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
    57a5:	00 
    57a6:	4c 8d 6c 24 40       	lea    0x40(%rsp),%r13
    57ab:	48 89 c7             	mov    %rax,%rdi
    57ae:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    57b3:	e8 c8 c8 ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
      seekg(off_type, ios_base::seekdir);
      ///@}

    protected:
      basic_istream()
      : _M_gcount(streamsize(0))
    57b8:	4c 8b 3d d1 25 00 00 	mov    0x25d1(%rip),%r15        # 7d90 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
      : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
    57bf:	31 d2                	xor    %edx,%edx
      { this->init(0); }
    57c1:	31 f6                	xor    %esi,%esi
	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
    57c3:	48 8d 0d ee 23 00 00 	lea    0x23ee(%rip),%rcx        # 7bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    57ca:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
      : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
    57ce:	66 89 94 24 20 02 00 	mov    %dx,0x220(%rsp)
    57d5:	00 
	_M_streambuf(0), _M_ctype(0), _M_num_put(0), _M_num_get(0)
    57d6:	c5 fe 7f 84 24 28 02 	vmovdqu %ymm0,0x228(%rsp)
    57dd:	00 00 
      : _M_gcount(streamsize(0))
    57df:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    57e3:	48 89 8c 24 40 01 00 	mov    %rcx,0x140(%rsp)
    57ea:	00 
    57eb:	48 8b 0d a6 25 00 00 	mov    0x25a6(%rip),%rcx        # 7d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
      : ios_base(), _M_tie(0), _M_fill(char_type()), _M_fill_init(false), 
    57f2:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
    57f9:	00 00 00 00 00 
    57fe:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    5803:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    5808:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    580f:	00 00 
      { this->init(0); }
    5811:	49 8b 4f e8          	mov    -0x18(%r15),%rcx
    5815:	4c 01 e9             	add    %r13,%rcx
    5818:	48 89 cf             	mov    %rcx,%rdi
    581b:	c5 f8 77             	vzeroupper
    581e:	e8 8d c9 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
      : __istream_type(), _M_filebuf()
    5823:	48 8d 0d 56 24 00 00 	lea    0x2456(%rip),%rcx        # 7c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    582a:	4c 8d 74 24 50       	lea    0x50(%rsp),%r14
    582f:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
    5834:	4c 89 f7             	mov    %r14,%rdi
    5837:	48 83 c1 28          	add    $0x28,%rcx
    583b:	48 89 8c 24 40 01 00 	mov    %rcx,0x140(%rsp)
    5842:	00 
    5843:	4c 89 34 24          	mov    %r14,(%rsp)
    5847:	e8 14 c9 ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
	this->init(&_M_filebuf);
    584c:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    5851:	4c 89 f6             	mov    %r14,%rsi
    5854:	e8 57 c9 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
      { return open(__s.c_str(), __mode); }
    5859:	49 8b 34 24          	mov    (%r12),%rsi
    585d:	ba 08 00 00 00       	mov    $0x8,%edx
    5862:	4c 89 f7             	mov    %r14,%rdi
    5865:	e8 c6 c8 ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
	  this->setstate(ios_base::failbit);
    586a:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    586f:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    5873:	4c 01 ef             	add    %r13,%rdi
	if (!_M_filebuf.open(__s, __mode | ios_base::in))
    5876:	48 85 c0             	test   %rax,%rax
    5879:	0f 84 31 02 00 00    	je     5ab0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x330>
	  this->clear();
    587f:	31 f6                	xor    %esi,%esi
    5881:	e8 aa c9 ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
	: _M_start(), _M_finish(), _M_end_of_storage()
    5886:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
    588d:	00 
    588e:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    5892:	4c 8d 74 24 38       	lea    0x38(%rsp),%r14
    5897:	c5 fa 7f 03          	vmovdqu %xmm0,(%rbx)
      { return _M_extract(__f); }
    589b:	4c 89 f6             	mov    %r14,%rsi
    589e:	4c 89 ef             	mov    %r13,%rdi
    58a1:	e8 fa c9 ff ff       	call   22a0 <_ZNSi10_M_extractIfEERSiRT_@plt>
    58a6:	48 89 c7             	mov    %rax,%rdi
    58a9:	48 8d 74 24 3c       	lea    0x3c(%rsp),%rsi
    58ae:	e8 ed c9 ff ff       	call   22a0 <_ZNSi10_M_extractIfEERSiRT_@plt>
    while (fin >> re >> im)
    58b3:	48 8b 10             	mov    (%rax),%rdx
      { return _M_streambuf_state; }
    58b6:	48 8b 52 e8          	mov    -0x18(%rdx),%rdx
    58ba:	f6 44 10 20 05       	testb  $0x5,0x20(%rax,%rdx,1)
    58bf:	0f 85 4b 01 00 00    	jne    5a10 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x290>
      void
#endif
      vector<_Tp, _Alloc>::
      emplace_back(_Args&&... __args)
      {
	if (this->_M_impl._M_finish != this->_M_impl._M_end_of_storage)
    58c5:	48 8b 53 08          	mov    0x8(%rbx),%rdx
    58c9:	48 3b 53 10          	cmp    0x10(%rbx),%rdx
    58cd:	74 31                	je     5900 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x180>
      : _M_value{ __r, __i } { }
    58cf:	c5 fa 10 44 24 38    	vmovss 0x38(%rsp),%xmm0
	  {
	    _GLIBCXX_ASAN_ANNOTATE_GROW(1);
	    _Alloc_traits::construct(this->_M_impl, this->_M_impl._M_finish,
				     std::forward<_Args>(__args)...);
	    ++this->_M_impl._M_finish;
    58d5:	48 83 c2 08          	add    $0x8,%rdx
    58d9:	4c 89 f6             	mov    %r14,%rsi
    58dc:	4c 89 ef             	mov    %r13,%rdi
    58df:	c4 e3 79 21 44 24 3c 	vinsertps $0x10,0x3c(%rsp),%xmm0,%xmm0
    58e6:	10 
    58e7:	c5 f8 13 42 f8       	vmovlps %xmm0,-0x8(%rdx)
    58ec:	48 89 53 08          	mov    %rdx,0x8(%rbx)
    58f0:	e8 ab c9 ff ff       	call   22a0 <_ZNSi10_M_extractIfEERSiRT_@plt>
    58f5:	eb af                	jmp    58a6 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x126>
    58f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    58fe:	00 00 
	if (max_size() - size() < __n)
    5900:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    5907:	ff ff 0f 
    590a:	48 8b 0b             	mov    (%rbx),%rcx
      { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
    590d:	48 89 d6             	mov    %rdx,%rsi
    5910:	48 29 ce             	sub    %rcx,%rsi
    5913:	49 89 f4             	mov    %rsi,%r12
    5916:	49 c1 fc 03          	sar    $0x3,%r12
	if (max_size() - size() < __n)
    591a:	49 39 c4             	cmp    %rax,%r12
    591d:	0f 84 1c 02 00 00    	je     5b3f <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3bf>
      if (__a < __b)
    5923:	4d 85 e4             	test   %r12,%r12
    5926:	b8 01 00 00 00       	mov    $0x1,%eax
    592b:	49 0f 45 c4          	cmovne %r12,%rax
    592f:	49 01 c4             	add    %rax,%r12
    5932:	0f 82 88 01 00 00    	jb     5ac0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x340>
	return (__len < size() || __len > max_size()) ? max_size() : __len;
    5938:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    593f:	ff ff 0f 
    5942:	49 39 c4             	cmp    %rax,%r12
    5945:	4c 0f 47 e0          	cmova  %rax,%r12
	return static_cast<_Tp*>(_GLIBCXX_OPERATOR_NEW(__n * sizeof(_Tp)));
    5949:	49 c1 e4 03          	shl    $0x3,%r12
    594d:	4c 89 e7             	mov    %r12,%rdi
    5950:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
    5955:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    595a:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
    595f:	e8 dc c7 ff ff       	call   2140 <_Znwm@plt>
    5964:	c5 fa 10 44 24 38    	vmovss 0x38(%rsp),%xmm0
    596a:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    596f:	49 89 c0             	mov    %rax,%r8
    5972:	c4 e3 79 21 44 24 3c 	vinsertps $0x10,0x3c(%rsp),%xmm0,%xmm0
    5979:	10 
      typedef typename iterator_traits<_ForwardIterator>::value_type
	_ValueType2;
      static_assert(std::is_same<_ValueType, _ValueType2>::value,
	  "relocation is only possible for values of the same type");
      _ForwardIterator __cur = __result;
      for (; __first != __last; ++__first, (void)++__cur)
    597a:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    597f:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    5984:	c5 f8 13 04 30       	vmovlps %xmm0,(%rax,%rsi,1)
    5989:	48 39 ca             	cmp    %rcx,%rdx
    598c:	74 32                	je     59c0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x240>
    598e:	48 29 ca             	sub    %rcx,%rdx
    5991:	48 89 ce             	mov    %rcx,%rsi
    5994:	48 01 c2             	add    %rax,%rdx
    5997:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    599e:	00 00 
      template<typename _Up, typename... _Args>
	__attribute__((__always_inline__))
	void
	construct(_Up* __p, _Args&&... __args)
	noexcept(__is_nothrow_new_constructible<_Up, _Args...>)
	{ ::new((void *)__p) _Up(std::forward<_Args>(__args)...); }
    59a0:	c5 fa 10 06          	vmovss (%rsi),%xmm0
    59a4:	48 83 c0 08          	add    $0x8,%rax
    59a8:	48 83 c6 08          	add    $0x8,%rsi
    59ac:	c5 fa 11 40 f8       	vmovss %xmm0,-0x8(%rax)
    59b1:	c5 fa 10 46 fc       	vmovss -0x4(%rsi),%xmm0
    59b6:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
    59bb:	48 39 d0             	cmp    %rdx,%rax
    59be:	75 e0                	jne    59a0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x220>
	if _GLIBCXX17_CONSTEXPR (_S_use_relocate())
	  {
	    // Relocation cannot throw.
	    __new_finish = _S_relocate(__old_start, __old_finish,
				       __new_start, _M_get_Tp_allocator());
	    ++__new_finish;
    59c0:	48 83 c0 08          	add    $0x8,%rax
    59c4:	c4 c1 f9 6e c8       	vmovq  %r8,%xmm1
    59c9:	c4 e3 f1 22 c0 01    	vpinsrq $0x1,%rax,%xmm1,%xmm0
	  if (_M_storage)
    59cf:	48 85 c9             	test   %rcx,%rcx
    59d2:	74 25                	je     59f9 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x279>
	    // New storage has been fully initialized, destroy the old elements.
	    __guard_elts._M_first = __old_start;
	    __guard_elts._M_last = __old_finish;
	  }
	__guard._M_storage = __old_start;
	__guard._M_len = this->_M_impl._M_end_of_storage - __old_start;
    59d4:	48 8b 73 10          	mov    0x10(%rbx),%rsi
	_GLIBCXX_OPERATOR_DELETE(_GLIBCXX_SIZED_DEALLOC(__p, __n));
    59d8:	48 89 cf             	mov    %rcx,%rdi
    59db:	4c 89 44 24 20       	mov    %r8,0x20(%rsp)
    59e0:	c5 f9 7f 44 24 10    	vmovdqa %xmm0,0x10(%rsp)
    59e6:	48 29 ce             	sub    %rcx,%rsi
    59e9:	e8 62 c7 ff ff       	call   2150 <_ZdlPvm@plt>
    59ee:	4c 8b 44 24 20       	mov    0x20(%rsp),%r8
    59f3:	c5 f9 6f 44 24 10    	vmovdqa 0x10(%rsp),%xmm0
      // deallocate should be called before assignments to _M_impl,
      // to avoid call-clobbering

      this->_M_impl._M_start = __new_start;
      this->_M_impl._M_finish = __new_finish;
      this->_M_impl._M_end_of_storage = __new_start + __len;
    59f9:	4d 01 e0             	add    %r12,%r8
      this->_M_impl._M_start = __new_start;
    59fc:	c5 fa 7f 03          	vmovdqu %xmm0,(%rbx)
      this->_M_impl._M_end_of_storage = __new_start + __len;
    5a00:	4c 89 43 10          	mov    %r8,0x10(%rbx)
    }
    5a04:	e9 92 fe ff ff       	jmp    589b <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x11b>
    5a09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
      { }
    5a10:	48 8d 05 69 22 00 00 	lea    0x2269(%rip),%rax        # 7c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
	  { this->close(); }
    5a17:	48 8b 3c 24          	mov    (%rsp),%rdi
      { }
    5a1b:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    5a20:	48 83 c0 28          	add    $0x28,%rax
    5a24:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    5a2b:	00 
      }
    5a2c:	48 8d 05 95 22 00 00 	lea    0x2295(%rip),%rax        # 7cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    5a33:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
	  { this->close(); }
    5a38:	e8 13 c6 ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
      }
    5a3d:	48 8d bc 24 b8 00 00 	lea    0xb8(%rsp),%rdi
    5a44:	00 
    5a45:	e8 26 c8 ff ff       	call   2270 <_ZNSt12__basic_fileIcED1Ev@plt>
    5a4a:	48 8d 05 87 21 00 00 	lea    0x2187(%rip),%rax        # 7bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    5a51:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
    5a58:	00 
    5a59:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    5a5e:	e8 5d c7 ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
      { _M_gcount = streamsize(0); }
    5a63:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    5a67:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    5a6c:	48 8b 0d 25 23 00 00 	mov    0x2325(%rip),%rcx        # 7d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
      ~basic_ios() { }
    5a73:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    5a78:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    5a7d:	48 8d 05 34 21 00 00 	lea    0x2134(%rip),%rax        # 7bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    5a84:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    5a8b:	00 
    5a8c:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    5a93:	00 00 
    5a95:	e8 f6 c5 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
}
    5a9a:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    5a9e:	48 89 d8             	mov    %rbx,%rax
    5aa1:	5b                   	pop    %rbx
    5aa2:	41 5c                	pop    %r12
    5aa4:	41 5d                	pop    %r13
    5aa6:	41 5e                	pop    %r14
    5aa8:	41 5f                	pop    %r15
    5aaa:	5d                   	pop    %rbp
    5aab:	c3                   	ret
    5aac:	0f 1f 40 00          	nopl   0x0(%rax)
    5ab0:	8b 77 20             	mov    0x20(%rdi),%esi
    5ab3:	83 ce 04             	or     $0x4,%esi
      { this->clear(this->rdstate() | __state); }
    5ab6:	e8 75 c7 ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    5abb:	e9 c6 fd ff ff       	jmp    5886 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x106>
    5ac0:	49 bc f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%r12
    5ac7:	ff ff 7f 
    5aca:	e9 7e fe ff ff       	jmp    594d <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1cd>
    5acf:	48 89 c3             	mov    %rax,%rbx
    5ad2:	eb 11                	jmp    5ae5 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x365>
      ~basic_ios() { }
    5ad4:	48 89 c3             	mov    %rax,%rbx
    5ad7:	eb 28                	jmp    5b01 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x381>
      }
    5ad9:	48 8b 3c 24          	mov    (%rsp),%rdi
    5add:	c5 f8 77             	vzeroupper
    5ae0:	e8 fb c6 ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
    5ae5:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    5ae9:	48 8b 0d a8 22 00 00 	mov    0x22a8(%rip),%rcx        # 7d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    5af0:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    5af5:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    5afa:	31 c0                	xor    %eax,%eax
    5afc:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    5b01:	48 8d 05 b0 20 00 00 	lea    0x20b0(%rip),%rax        # 7bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    5b08:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    5b0d:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    5b14:	00 
    5b15:	c5 f8 77             	vzeroupper
    5b18:	e8 73 c5 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    5b1d:	48 89 df             	mov    %rbx,%rdi
    5b20:	e8 1b c7 ff ff       	call   2240 <_Unwind_Resume@plt>
    5b25:	48 89 c3             	mov    %rax,%rbx
    5b28:	eb af                	jmp    5ad9 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x359>
	__catch(...)
    5b2a:	48 89 c7             	mov    %rax,%rdi
    5b2d:	c5 f8 77             	vzeroupper
    5b30:	e8 6b c5 ff ff       	call   20a0 <__cxa_begin_catch@plt>
    5b35:	e8 e6 c6 ff ff       	call   2220 <__cxa_end_catch@plt>
    5b3a:	e9 fe fe ff ff       	jmp    5a3d <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2bd>
	  __throw_length_error(__N(__s));
    5b3f:	48 8d 3d ce 04 00 00 	lea    0x4ce(%rip),%rdi        # 6014 <_IO_stdin_used+0x14>
    5b46:	e8 75 c5 ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
		      _M_impl._M_end_of_storage - _M_impl._M_start);
    5b4b:	49 89 c4             	mov    %rax,%r12
    5b4e:	48 8b 3b             	mov    (%rbx),%rdi
    5b51:	48 8b 73 10          	mov    0x10(%rbx),%rsi
    5b55:	48 29 fe             	sub    %rdi,%rsi
	if (__p)
    5b58:	48 85 ff             	test   %rdi,%rdi
    5b5b:	74 18                	je     5b75 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3f5>
    5b5d:	c5 f8 77             	vzeroupper
    5b60:	e8 eb c5 ff ff       	call   2150 <_ZdlPvm@plt>
    5b65:	4c 89 ef             	mov    %r13,%rdi
    5b68:	e8 03 c5 ff ff       	call   2070 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@plt>
    5b6d:	4c 89 e7             	mov    %r12,%rdi
    5b70:	e8 cb c6 ff ff       	call   2240 <_Unwind_Resume@plt>
    5b75:	c5 f8 77             	vzeroupper
    5b78:	eb eb                	jmp    5b65 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3e5>

Disassembly of section .fini:

0000000000005b7c <_fini>:
    5b7c:	48 83 ec 08          	sub    $0x8,%rsp
    5b80:	48 83 c4 08          	add    $0x8,%rsp
    5b84:	c3                   	ret
