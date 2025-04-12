
./fft_cpu:     file format elf64-x86-64


Disassembly of section .init:

0000000000002000 <_init>:
    2000:	48 83 ec 08          	sub    $0x8,%rsp
    2004:	48 8b 05 cd 3f 00 00 	mov    0x3fcd(%rip),%rax        # 5fd8 <__gmon_start__@Base>
    200b:	48 85 c0             	test   %rax,%rax
    200e:	74 02                	je     2012 <_init+0x12>
    2010:	ff d0                	call   *%rax
    2012:	48 83 c4 08          	add    $0x8,%rsp
    2016:	c3                   	ret

Disassembly of section .plt:

0000000000002020 <_ZNSo3putEc@plt-0x10>:
    2020:	ff 35 ca 3f 00 00    	push   0x3fca(%rip)        # 5ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
    2026:	ff 25 cc 3f 00 00    	jmp    *0x3fcc(%rip)        # 5ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
    202c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002030 <_ZNSo3putEc@plt>:
    2030:	ff 25 ca 3f 00 00    	jmp    *0x3fca(%rip)        # 6000 <_ZNSo3putEc@GLIBCXX_3.4>
    2036:	68 00 00 00 00       	push   $0x0
    203b:	e9 e0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>:
    2040:	ff 25 c2 3f 00 00    	jmp    *0x3fc2(%rip)        # 6008 <_ZNSt6chrono3_V212system_clock3nowEv@GLIBCXX_3.4.19>
    2046:	68 01 00 00 00       	push   $0x1
    204b:	e9 d0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>:
    2050:	ff 25 ba 3f 00 00    	jmp    *0x3fba(%rip)        # 6010 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@GLIBCXX_3.4>
    2056:	68 02 00 00 00       	push   $0x2
    205b:	e9 c0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002060 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@plt>:
    2060:	ff 25 b2 3f 00 00    	jmp    *0x3fb2(%rip)        # 6018 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@GLIBCXX_3.4.21>
    2066:	68 03 00 00 00       	push   $0x3
    206b:	e9 b0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002070 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@plt>:
    2070:	ff 25 aa 3f 00 00    	jmp    *0x3faa(%rip)        # 6020 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    2076:	68 04 00 00 00       	push   $0x4
    207b:	e9 a0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002080 <_ZNSt8ios_baseC2Ev@plt>:
    2080:	ff 25 a2 3f 00 00    	jmp    *0x3fa2(%rip)        # 6028 <_ZNSt8ios_baseC2Ev@GLIBCXX_3.4>
    2086:	68 05 00 00 00       	push   $0x5
    208b:	e9 90 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002090 <_ZNSt8ios_baseD2Ev@plt>:
    2090:	ff 25 9a 3f 00 00    	jmp    *0x3f9a(%rip)        # 6030 <_ZNSt8ios_baseD2Ev@GLIBCXX_3.4>
    2096:	68 06 00 00 00       	push   $0x6
    209b:	e9 80 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020a0 <__cxa_begin_catch@plt>:
    20a0:	ff 25 92 3f 00 00    	jmp    *0x3f92(%rip)        # 6038 <__cxa_begin_catch@CXXABI_1.3>
    20a6:	68 07 00 00 00       	push   $0x7
    20ab:	e9 70 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020b0 <strlen@plt>:
    20b0:	ff 25 8a 3f 00 00    	jmp    *0x3f8a(%rip)        # 6040 <strlen@GLIBC_2.2.5>
    20b6:	68 08 00 00 00       	push   $0x8
    20bb:	e9 60 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020c0 <_ZSt20__throw_length_errorPKc@plt>:
    20c0:	ff 25 82 3f 00 00    	jmp    *0x3f82(%rip)        # 6048 <_ZSt20__throw_length_errorPKc@GLIBCXX_3.4>
    20c6:	68 09 00 00 00       	push   $0x9
    20cb:	e9 50 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>:
    20d0:	ff 25 7a 3f 00 00    	jmp    *0x3f7a(%rip)        # 6050 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@GLIBCXX_3.4.21>
    20d6:	68 0a 00 00 00       	push   $0xa
    20db:	e9 40 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020e0 <_ZNSo5flushEv@plt>:
    20e0:	ff 25 72 3f 00 00    	jmp    *0x3f72(%rip)        # 6058 <_ZNSo5flushEv@GLIBCXX_3.4>
    20e6:	68 0b 00 00 00       	push   $0xb
    20eb:	e9 30 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020f0 <_ZSt19__throw_logic_errorPKc@plt>:
    20f0:	ff 25 6a 3f 00 00    	jmp    *0x3f6a(%rip)        # 6060 <_ZSt19__throw_logic_errorPKc@GLIBCXX_3.4>
    20f6:	68 0c 00 00 00       	push   $0xc
    20fb:	e9 20 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002100 <cosf@plt>:
    2100:	ff 25 62 3f 00 00    	jmp    *0x3f62(%rip)        # 6068 <cosf@GLIBC_2.2.5>
    2106:	68 0d 00 00 00       	push   $0xd
    210b:	e9 10 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002110 <memcpy@plt>:
    2110:	ff 25 5a 3f 00 00    	jmp    *0x3f5a(%rip)        # 6070 <memcpy@GLIBC_2.14>
    2116:	68 0e 00 00 00       	push   $0xe
    211b:	e9 00 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002120 <sinf@plt>:
    2120:	ff 25 52 3f 00 00    	jmp    *0x3f52(%rip)        # 6078 <sinf@GLIBC_2.2.5>
    2126:	68 0f 00 00 00       	push   $0xf
    212b:	e9 f0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>:
    2130:	ff 25 4a 3f 00 00    	jmp    *0x3f4a(%rip)        # 6080 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@GLIBCXX_3.4>
    2136:	68 10 00 00 00       	push   $0x10
    213b:	e9 e0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002140 <_Znwm@plt>:
    2140:	ff 25 42 3f 00 00    	jmp    *0x3f42(%rip)        # 6088 <_Znwm@GLIBCXX_3.4>
    2146:	68 11 00 00 00       	push   $0x11
    214b:	e9 d0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002150 <_ZdlPvm@plt>:
    2150:	ff 25 3a 3f 00 00    	jmp    *0x3f3a(%rip)        # 6090 <_ZdlPvm@CXXABI_1.3.9>
    2156:	68 12 00 00 00       	push   $0x12
    215b:	e9 c0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>:
    2160:	ff 25 32 3f 00 00    	jmp    *0x3f32(%rip)        # 6098 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@GLIBCXX_3.4>
    2166:	68 13 00 00 00       	push   $0x13
    216b:	e9 b0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
    2170:	ff 25 2a 3f 00 00    	jmp    *0x3f2a(%rip)        # 60a0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
    2176:	68 14 00 00 00       	push   $0x14
    217b:	e9 a0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>:
    2180:	ff 25 22 3f 00 00    	jmp    *0x3f22(%rip)        # 60a8 <_ZNKSt5ctypeIcE13_M_widen_initEv@GLIBCXX_3.4.11>
    2186:	68 15 00 00 00       	push   $0x15
    218b:	e9 90 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>:
    2190:	ff 25 1a 3f 00 00    	jmp    *0x3f1a(%rip)        # 60b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@GLIBCXX_3.4.21>
    2196:	68 16 00 00 00       	push   $0x16
    219b:	e9 80 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021a0 <_ZSt16__throw_bad_castv@plt>:
    21a0:	ff 25 12 3f 00 00    	jmp    *0x3f12(%rip)        # 60b8 <_ZSt16__throw_bad_castv@GLIBCXX_3.4>
    21a6:	68 17 00 00 00       	push   $0x17
    21ab:	e9 70 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>:
    21b0:	ff 25 0a 3f 00 00    	jmp    *0x3f0a(%rip)        # 60c0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@GLIBCXX_3.4>
    21b6:	68 18 00 00 00       	push   $0x18
    21bb:	e9 60 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021c0 <_ZNSt6localeD1Ev@plt>:
    21c0:	ff 25 02 3f 00 00    	jmp    *0x3f02(%rip)        # 60c8 <_ZNSt6localeD1Ev@GLIBCXX_3.4>
    21c6:	68 19 00 00 00       	push   $0x19
    21cb:	e9 50 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>:
    21d0:	ff 25 fa 3e 00 00    	jmp    *0x3efa(%rip)        # 60d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    21d6:	68 1a 00 00 00       	push   $0x1a
    21db:	e9 40 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>:
    21e0:	ff 25 f2 3e 00 00    	jmp    *0x3ef2(%rip)        # 60d8 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    21e6:	68 1b 00 00 00       	push   $0x1b
    21eb:	e9 30 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@plt>:
    21f0:	ff 25 ea 3e 00 00    	jmp    *0x3eea(%rip)        # 60e0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@GLIBCXX_3.4.21>
    21f6:	68 1c 00 00 00       	push   $0x1c
    21fb:	e9 20 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002200 <_ZNSo9_M_insertIdEERSoT_@plt>:
    2200:	ff 25 e2 3e 00 00    	jmp    *0x3ee2(%rip)        # 60e8 <_ZNSo9_M_insertIdEERSoT_@GLIBCXX_3.4.9>
    2206:	68 1d 00 00 00       	push   $0x1d
    220b:	e9 10 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002210 <memmove@plt>:
    2210:	ff 25 da 3e 00 00    	jmp    *0x3eda(%rip)        # 60f0 <memmove@GLIBC_2.2.5>
    2216:	68 1e 00 00 00       	push   $0x1e
    221b:	e9 00 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002220 <__cxa_end_catch@plt>:
    2220:	ff 25 d2 3e 00 00    	jmp    *0x3ed2(%rip)        # 60f8 <__cxa_end_catch@CXXABI_1.3>
    2226:	68 1f 00 00 00       	push   $0x1f
    222b:	e9 f0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>:
    2230:	ff 25 ca 3e 00 00    	jmp    *0x3eca(%rip)        # 6100 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@GLIBCXX_3.4>
    2236:	68 20 00 00 00       	push   $0x20
    223b:	e9 e0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002240 <_ZNSolsEi@plt>:
    2240:	ff 25 c2 3e 00 00    	jmp    *0x3ec2(%rip)        # 6108 <_ZNSolsEi@GLIBCXX_3.4>
    2246:	68 21 00 00 00       	push   $0x21
    224b:	e9 d0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002250 <_Unwind_Resume@plt>:
    2250:	ff 25 ba 3e 00 00    	jmp    *0x3eba(%rip)        # 6110 <_Unwind_Resume@GCC_3.0>
    2256:	68 22 00 00 00       	push   $0x22
    225b:	e9 c0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002260 <log2@plt>:
    2260:	ff 25 b2 3e 00 00    	jmp    *0x3eb2(%rip)        # 6118 <log2@GLIBC_2.29>
    2266:	68 23 00 00 00       	push   $0x23
    226b:	e9 b0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002270 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>:
    2270:	ff 25 aa 3e 00 00    	jmp    *0x3eaa(%rip)        # 6120 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@GLIBCXX_3.4.21>
    2276:	68 24 00 00 00       	push   $0x24
    227b:	e9 a0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002280 <_ZNSt12__basic_fileIcED1Ev@plt>:
    2280:	ff 25 a2 3e 00 00    	jmp    *0x3ea2(%rip)        # 6128 <_ZNSt12__basic_fileIcED1Ev@GLIBCXX_3.4>
    2286:	68 25 00 00 00       	push   $0x25
    228b:	e9 90 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002290 <__mulsc3@plt>:
    2290:	ff 25 9a 3e 00 00    	jmp    *0x3e9a(%rip)        # 6130 <__mulsc3@GCC_4.0.0>
    2296:	68 26 00 00 00       	push   $0x26
    229b:	e9 80 fd ff ff       	jmp    2020 <_init+0x20>

00000000000022a0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>:
    22a0:	ff 25 92 3e 00 00    	jmp    *0x3e92(%rip)        # 6138 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@GLIBCXX_3.4.21>
    22a6:	68 27 00 00 00       	push   $0x27
    22ab:	e9 70 fd ff ff       	jmp    2020 <_init+0x20>

00000000000022b0 <_ZNSi10_M_extractIfEERSiRT_@plt>:
    22b0:	ff 25 8a 3e 00 00    	jmp    *0x3e8a(%rip)        # 6140 <_ZNSi10_M_extractIfEERSiRT_@GLIBCXX_3.4.9>
    22b6:	68 28 00 00 00       	push   $0x28
    22bb:	e9 60 fd ff ff       	jmp    2020 <_init+0x20>

Disassembly of section .plt.got:

00000000000022c0 <__cxa_finalize@plt>:
    22c0:	ff 25 fa 3c 00 00    	jmp    *0x3cfa(%rip)        # 5fc0 <__cxa_finalize@GLIBC_2.2.5>
    22c6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000000022e0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0.cold>:
    22e0:	e8 bb fe ff ff       	call   21a0 <_ZSt16__throw_bad_castv@plt>

00000000000022e5 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_.cold>:
    22e5:	e8 b6 fe ff ff       	call   21a0 <_ZSt16__throw_bad_castv@plt>

00000000000022ea <main.cold>:
    22ea:	48 8d 3d 7f 1d 00 00 	lea    0x1d7f(%rip),%rdi        # 4070 <_IO_stdin_used+0x70>
    22f1:	e8 fa fd ff ff       	call   20f0 <_ZSt19__throw_logic_errorPKc@plt>
    22f6:	49 89 c4             	mov    %rax,%r12
    22f9:	c5 f8 77             	vzeroupper
    22fc:	48 8b b5 08 fd ff ff 	mov    -0x2f8(%rbp),%rsi
    2303:	48 8b 85 30 fd ff ff 	mov    -0x2d0(%rbp),%rax
    230a:	48 29 c6             	sub    %rax,%rsi
    230d:	48 85 c0             	test   %rax,%rax
    2310:	74 08                	je     231a <main.cold+0x30>
    2312:	48 89 c7             	mov    %rax,%rdi
    2315:	e8 36 fe ff ff       	call   2150 <_ZdlPvm@plt>
    231a:	4c 89 e7             	mov    %r12,%rdi
    231d:	e8 2e ff ff ff       	call   2250 <_Unwind_Resume@plt>
    2322:	4c 89 e7             	mov    %r12,%rdi
    2325:	c5 f8 77             	vzeroupper
    2328:	4d 89 ec             	mov    %r13,%r12
    232b:	e8 60 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2330:	48 89 df             	mov    %rbx,%rdi
    2333:	e8 58 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2338:	48 83 bd 40 fd ff ff 	cmpq   $0x0,-0x2c0(%rbp)
    233f:	00 
    2340:	74 ba                	je     22fc <main.cold+0x12>
    2342:	48 8b b5 28 fd ff ff 	mov    -0x2d8(%rbp),%rsi
    2349:	48 8b bd 40 fd ff ff 	mov    -0x2c0(%rbp),%rdi
    2350:	e8 fb fd ff ff       	call   2150 <_ZdlPvm@plt>
    2355:	eb a5                	jmp    22fc <main.cold+0x12>
    2357:	4c 89 e7             	mov    %r12,%rdi
    235a:	c5 f8 77             	vzeroupper
    235d:	49 89 dc             	mov    %rbx,%r12
    2360:	e8 6b fe ff ff       	call   21d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    2365:	48 8b bd 20 fd ff ff 	mov    -0x2e0(%rbp),%rdi
    236c:	e8 1f fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2371:	eb c5                	jmp    2338 <main.cold+0x4e>
    2373:	c5 f8 77             	vzeroupper
    2376:	e8 25 fd ff ff       	call   20a0 <__cxa_begin_catch@plt>
    237b:	e8 a0 fe ff ff       	call   2220 <__cxa_end_catch@plt>
    2380:	e9 83 06 00 00       	jmp    2a08 <main+0x5e8>
    2385:	48 8d 3d 1c 1d 00 00 	lea    0x1d1c(%rip),%rdi        # 40a8 <_IO_stdin_used+0xa8>
    238c:	e8 2f fd ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
    2391:	4c 89 e7             	mov    %r12,%rdi
    2394:	c5 f8 77             	vzeroupper
    2397:	e8 f4 fd ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    239c:	48 89 df             	mov    %rbx,%rdi
    239f:	e8 ac fe ff ff       	call   2250 <_Unwind_Resume@plt>
    23a4:	48 8b bd 18 fd ff ff 	mov    -0x2e8(%rbp),%rdi
    23ab:	c5 f8 77             	vzeroupper
    23ae:	e8 2d fe ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
    23b3:	48 8b 05 96 38 00 00 	mov    0x3896(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    23ba:	48 8b 0d 97 38 00 00 	mov    0x3897(%rip),%rcx        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    23c1:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    23c8:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    23cc:	48 89 8c 05 d0 fd ff 	mov    %rcx,-0x230(%rbp,%rax,1)
    23d3:	ff 
    23d4:	48 8b bd 10 fd ff ff 	mov    -0x2f0(%rbp),%rdi
    23db:	48 8d 05 d6 37 00 00 	lea    0x37d6(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    23e2:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    23e9:	e8 a2 fc ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    23ee:	e9 72 ff ff ff       	jmp    2365 <main.cold+0x7b>
    23f3:	48 89 df             	mov    %rbx,%rdi
    23f6:	c5 f8 77             	vzeroupper
    23f9:	e8 92 fd ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    23fe:	e9 35 ff ff ff       	jmp    2338 <main.cold+0x4e>
    2403:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    240a:	00 00 00 
    240d:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2414:	00 00 00 
    2417:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    241e:	00 00 

0000000000002420 <main>:
    2420:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    2425:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    2429:	41 ff 72 f8          	push   -0x8(%r10)
    242d:	55                   	push   %rbp
    242e:	48 89 e5             	mov    %rsp,%rbp
    2431:	41 57                	push   %r15
    2433:	41 56                	push   %r14
    2435:	41 55                	push   %r13
    2437:	41 54                	push   %r12
    2439:	41 52                	push   %r10
    243b:	53                   	push   %rbx
    243c:	48 81 ec e0 02 00 00 	sub    $0x2e0,%rsp
    2443:	83 ff 01             	cmp    $0x1,%edi
    2446:	0f 8e d7 06 00 00    	jle    2b23 <main+0x703>
    244c:	4c 8b 6e 08          	mov    0x8(%rsi),%r13
    2450:	4c 8d b5 e0 fd ff ff 	lea    -0x220(%rbp),%r14
    2457:	4c 8d a5 d0 fd ff ff 	lea    -0x230(%rbp),%r12
    245e:	4c 89 b5 d0 fd ff ff 	mov    %r14,-0x230(%rbp)
    2465:	4d 85 ed             	test   %r13,%r13
    2468:	0f 84 7c fe ff ff    	je     22ea <main.cold>
    246e:	4c 89 ef             	mov    %r13,%rdi
    2471:	e8 3a fc ff ff       	call   20b0 <strlen@plt>
    2476:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
    247d:	48 89 c3             	mov    %rax,%rbx
    2480:	48 83 f8 0f          	cmp    $0xf,%rax
    2484:	0f 87 3d 06 00 00    	ja     2ac7 <main+0x6a7>
    248a:	48 83 f8 01          	cmp    $0x1,%rax
    248e:	0f 85 bc 06 00 00    	jne    2b50 <main+0x730>
    2494:	41 0f b6 45 00       	movzbl 0x0(%r13),%eax
    2499:	88 85 e0 fd ff ff    	mov    %al,-0x220(%rbp)
    249f:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    24a6:	48 8b 95 d0 fd ff ff 	mov    -0x230(%rbp),%rdx
    24ad:	4c 8d b5 50 fd ff ff 	lea    -0x2b0(%rbp),%r14
    24b4:	4c 89 e6             	mov    %r12,%rsi
    24b7:	4c 89 f7             	mov    %r14,%rdi
    24ba:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
    24c1:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
    24c5:	e8 56 12 00 00       	call   3720 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>
    24ca:	48 8b 85 60 fd ff ff 	mov    -0x2a0(%rbp),%rax
    24d1:	4c 89 e7             	mov    %r12,%rdi
    24d4:	48 89 85 08 fd ff ff 	mov    %rax,-0x2f8(%rbp)
    24db:	e8 b0 fc ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    24e0:	48 8b 8d 50 fd ff ff 	mov    -0x2b0(%rbp),%rcx
    24e7:	48 8b 85 58 fd ff ff 	mov    -0x2a8(%rbp),%rax
    24ee:	48 29 c8             	sub    %rcx,%rax
    24f1:	48 89 8d 30 fd ff ff 	mov    %rcx,-0x2d0(%rbp)
    24f8:	48 89 c3             	mov    %rax,%rbx
    24fb:	48 89 85 28 fd ff ff 	mov    %rax,-0x2d8(%rbp)
    2502:	48 89 c1             	mov    %rax,%rcx
    2505:	48 89 85 38 fd ff ff 	mov    %rax,-0x2c8(%rbp)
    250c:	48 c1 fb 03          	sar    $0x3,%rbx
    2510:	48 b8 f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%rax
    2517:	ff ff 7f 
    251a:	49 89 df             	mov    %rbx,%r15
    251d:	48 39 c8             	cmp    %rcx,%rax
    2520:	0f 82 5f fe ff ff    	jb     2385 <main.cold+0x9b>
    2526:	48 c7 85 78 fd ff ff 	movq   $0x0,-0x288(%rbp)
    252d:	00 00 00 00 
    2531:	48 85 db             	test   %rbx,%rbx
    2534:	0f 84 c6 05 00 00    	je     2b00 <main+0x6e0>
    253a:	4c 8b ad 28 fd ff ff 	mov    -0x2d8(%rbp),%r13
    2541:	4c 89 ef             	mov    %r13,%rdi
    2544:	e8 f7 fb ff ff       	call   2140 <_Znwm@plt>
    2549:	4a 8d 3c 28          	lea    (%rax,%r13,1),%rdi
    254d:	48 89 85 40 fd ff ff 	mov    %rax,-0x2c0(%rbp)
    2554:	48 89 c1             	mov    %rax,%rcx
    2557:	48 8d 43 ff          	lea    -0x1(%rbx),%rax
    255b:	48 89 bd 20 fd ff ff 	mov    %rdi,-0x2e0(%rbp)
    2562:	48 83 f8 02          	cmp    $0x2,%rax
    2566:	0f 86 1c 06 00 00    	jbe    2b88 <main+0x768>
    256c:	48 89 de             	mov    %rbx,%rsi
    256f:	48 89 c8             	mov    %rcx,%rax
    2572:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    2576:	48 c1 ee 02          	shr    $0x2,%rsi
    257a:	48 c1 e6 05          	shl    $0x5,%rsi
    257e:	48 8d 14 0e          	lea    (%rsi,%rcx,1),%rdx
    2582:	40 80 e6 20          	and    $0x20,%sil
    2586:	74 18                	je     25a0 <main+0x180>
    2588:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    258f:	c5 fc 11 00          	vmovups %ymm0,(%rax)
    2593:	48 83 c0 20          	add    $0x20,%rax
    2597:	48 39 d0             	cmp    %rdx,%rax
    259a:	74 16                	je     25b2 <main+0x192>
    259c:	0f 1f 40 00          	nopl   0x0(%rax)
    25a0:	c5 fc 11 00          	vmovups %ymm0,(%rax)
    25a4:	48 83 c0 40          	add    $0x40,%rax
    25a8:	c5 fc 11 40 e0       	vmovups %ymm0,-0x20(%rax)
    25ad:	48 39 d0             	cmp    %rdx,%rax
    25b0:	75 ee                	jne    25a0 <main+0x180>
    25b2:	48 8b 8d 40 fd ff ff 	mov    -0x2c0(%rbp),%rcx
    25b9:	48 89 d8             	mov    %rbx,%rax
    25bc:	48 83 e0 fc          	and    $0xfffffffffffffffc,%rax
    25c0:	48 8d 34 c1          	lea    (%rcx,%rax,8),%rsi
    25c4:	48 39 c3             	cmp    %rax,%rbx
    25c7:	0f 84 b3 05 00 00    	je     2b80 <main+0x760>
    25cd:	c5 f8 77             	vzeroupper
    25d0:	48 89 da             	mov    %rbx,%rdx
    25d3:	48 29 c2             	sub    %rax,%rdx
    25d6:	48 83 fa 01          	cmp    $0x1,%rdx
    25da:	74 1d                	je     25f9 <main+0x1d9>
    25dc:	48 8b 8d 40 fd ff ff 	mov    -0x2c0(%rbp),%rcx
    25e3:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
    25e7:	c5 f8 11 04 c1       	vmovups %xmm0,(%rcx,%rax,8)
    25ec:	f6 c2 01             	test   $0x1,%dl
    25ef:	74 0d                	je     25fe <main+0x1de>
    25f1:	48 83 e2 fe          	and    $0xfffffffffffffffe,%rdx
    25f5:	48 8d 34 d6          	lea    (%rsi,%rdx,8),%rsi
    25f9:	31 c9                	xor    %ecx,%ecx
    25fb:	48 89 0e             	mov    %rcx,(%rsi)
    25fe:	48 89 bd 48 fd ff ff 	mov    %rdi,-0x2b8(%rbp)
    2605:	48 8b 85 48 fd ff ff 	mov    -0x2b8(%rbp),%rax
    260c:	48 89 85 78 fd ff ff 	mov    %rax,-0x288(%rbp)
    2613:	e8 28 fa ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    2618:	48 89 df             	mov    %rbx,%rdi
    261b:	48 8d 95 70 fd ff ff 	lea    -0x290(%rbp),%rdx
    2622:	4c 89 f6             	mov    %r14,%rsi
    2625:	49 89 c5             	mov    %rax,%r13
    2628:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    262f:	48 89 85 70 fd ff ff 	mov    %rax,-0x290(%rbp)
    2636:	48 8b 85 20 fd ff ff 	mov    -0x2e0(%rbp),%rax
    263d:	48 89 85 80 fd ff ff 	mov    %rax,-0x280(%rbp)
    2644:	e8 67 0d 00 00       	call   33b0 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_>
    2649:	e8 f2 f9 ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    264e:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
    2652:	48 8d 3d 27 3b 00 00 	lea    0x3b27(%rip),%rdi        # 6180 <_ZSt4cout@GLIBCXX_3.4>
    2659:	4c 29 e8             	sub    %r13,%rax
    265c:	c4 e1 ea 2a c0       	vcvtsi2ss %rax,%xmm2,%xmm0
    2661:	c5 fa 5e 05 a7 19 00 	vdivss 0x19a7(%rip),%xmm0,%xmm0        # 4010 <_IO_stdin_used+0x10>
    2668:	00 
    2669:	c5 fa 5a c0          	vcvtss2sd %xmm0,%xmm0,%xmm0
    266d:	e8 8e fb ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2672:	48 89 c7             	mov    %rax,%rdi
    2675:	e8 86 06 00 00       	call   2d00 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    267a:	48 83 bd 38 fd ff ff 	cmpq   $0x48,-0x2c8(%rbp)
    2681:	48 
    2682:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    2688:	48 be 4b 59 86 38 d6 	movabs $0x346dc5d63886594b,%rsi
    268f:	c5 6d 34 
    2692:	77 40                	ja     26d4 <main+0x2b4>
    2694:	eb 47                	jmp    26dd <main+0x2bd>
    2696:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    269d:	00 00 00 
    26a0:	48 81 fb e7 03 00 00 	cmp    $0x3e7,%rbx
    26a7:	0f 86 c1 04 00 00    	jbe    2b6e <main+0x74e>
    26ad:	48 81 fb 0f 27 00 00 	cmp    $0x270f,%rbx
    26b4:	0f 86 bd 04 00 00    	jbe    2b77 <main+0x757>
    26ba:	48 89 d8             	mov    %rbx,%rax
    26bd:	41 83 c5 04          	add    $0x4,%r13d
    26c1:	48 f7 e6             	mul    %rsi
    26c4:	48 c1 ea 0b          	shr    $0xb,%rdx
    26c8:	48 81 fb 9f 86 01 00 	cmp    $0x1869f,%rbx
    26cf:	76 0c                	jbe    26dd <main+0x2bd>
    26d1:	48 89 d3             	mov    %rdx,%rbx
    26d4:	48 83 fb 63          	cmp    $0x63,%rbx
    26d8:	77 c6                	ja     26a0 <main+0x280>
    26da:	41 ff c5             	inc    %r13d
    26dd:	48 8d 9d b0 fd ff ff 	lea    -0x250(%rbp),%rbx
    26e4:	45 89 ee             	mov    %r13d,%r14d
    26e7:	48 8d 85 c0 fd ff ff 	lea    -0x240(%rbp),%rax
    26ee:	48 c7 85 b8 fd ff ff 	movq   $0x0,-0x248(%rbp)
    26f5:	00 00 00 00 
    26f9:	4c 89 f6             	mov    %r14,%rsi
    26fc:	48 89 df             	mov    %rbx,%rdi
    26ff:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
    2706:	c6 85 c0 fd ff ff 00 	movb   $0x0,-0x240(%rbp)
    270d:	e8 de fa ff ff       	call   21f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@plt>
    2712:	c5 fd 6f 05 06 1a 00 	vmovdqa 0x1a06(%rip),%ymm0        # 4120 <_IO_stdin_used+0x120>
    2719:	00 
    271a:	41 ff cd             	dec    %r13d
    271d:	48 81 bd 38 fd ff ff 	cmpq   $0x318,-0x2c8(%rbp)
    2724:	18 03 00 00 
    2728:	48 8b bd b0 fd ff ff 	mov    -0x250(%rbp),%rdi
    272f:	c5 fd 7f 85 d0 fd ff 	vmovdqa %ymm0,-0x230(%rbp)
    2736:	ff 
    2737:	c5 fd 6f 05 01 1a 00 	vmovdqa 0x1a01(%rip),%ymm0        # 4140 <_IO_stdin_used+0x140>
    273e:	00 
    273f:	c5 fd 7f 85 f0 fd ff 	vmovdqa %ymm0,-0x210(%rbp)
    2746:	ff 
    2747:	c5 fd 6f 05 11 1a 00 	vmovdqa 0x1a11(%rip),%ymm0        # 4160 <_IO_stdin_used+0x160>
    274e:	00 
    274f:	c5 fd 7f 85 10 fe ff 	vmovdqa %ymm0,-0x1f0(%rbp)
    2756:	ff 
    2757:	c5 fd 6f 05 21 1a 00 	vmovdqa 0x1a21(%rip),%ymm0        # 4180 <_IO_stdin_used+0x180>
    275e:	00 
    275f:	c5 fd 7f 85 30 fe ff 	vmovdqa %ymm0,-0x1d0(%rbp)
    2766:	ff 
    2767:	c5 fd 6f 05 31 1a 00 	vmovdqa 0x1a31(%rip),%ymm0        # 41a0 <_IO_stdin_used+0x1a0>
    276e:	00 
    276f:	c5 fd 7f 85 50 fe ff 	vmovdqa %ymm0,-0x1b0(%rbp)
    2776:	ff 
    2777:	c5 fd 6f 05 41 1a 00 	vmovdqa 0x1a41(%rip),%ymm0        # 41c0 <_IO_stdin_used+0x1c0>
    277e:	00 
    277f:	c5 fd 7f 85 70 fe ff 	vmovdqa %ymm0,-0x190(%rbp)
    2786:	ff 
    2787:	c5 f9 6f 05 71 19 00 	vmovdqa 0x1971(%rip),%xmm0        # 4100 <_IO_stdin_used+0x100>
    278e:	00 
    278f:	c5 fa 7f 85 89 fe ff 	vmovdqu %xmm0,-0x177(%rbp)
    2796:	ff 
    2797:	76 5f                	jbe    27f8 <main+0x3d8>
    2799:	48 be c3 f5 28 5c 8f 	movabs $0x28f5c28f5c28f5c3,%rsi
    27a0:	c2 f5 28 
    27a3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    27a8:	4c 89 fa             	mov    %r15,%rdx
    27ab:	48 c1 ea 02          	shr    $0x2,%rdx
    27af:	48 89 d0             	mov    %rdx,%rax
    27b2:	48 f7 e6             	mul    %rsi
    27b5:	4c 89 f8             	mov    %r15,%rax
    27b8:	48 c1 ea 02          	shr    $0x2,%rdx
    27bc:	48 6b ca 64          	imul   $0x64,%rdx,%rcx
    27c0:	48 29 c8             	sub    %rcx,%rax
    27c3:	4c 89 f9             	mov    %r15,%rcx
    27c6:	49 89 d7             	mov    %rdx,%r15
    27c9:	44 89 ea             	mov    %r13d,%edx
    27cc:	48 01 c0             	add    %rax,%rax
    27cf:	44 0f b6 84 05 d1 fd 	movzbl -0x22f(%rbp,%rax,1),%r8d
    27d6:	ff ff 
    27d8:	44 88 04 17          	mov    %r8b,(%rdi,%rdx,1)
    27dc:	0f b6 84 05 d0 fd ff 	movzbl -0x230(%rbp,%rax,1),%eax
    27e3:	ff 
    27e4:	41 8d 55 ff          	lea    -0x1(%r13),%edx
    27e8:	41 83 ed 02          	sub    $0x2,%r13d
    27ec:	88 04 17             	mov    %al,(%rdi,%rdx,1)
    27ef:	48 81 f9 0f 27 00 00 	cmp    $0x270f,%rcx
    27f6:	77 b0                	ja     27a8 <main+0x388>
    27f8:	41 8d 47 30          	lea    0x30(%r15),%eax
    27fc:	49 83 ff 09          	cmp    $0x9,%r15
    2800:	76 17                	jbe    2819 <main+0x3f9>
    2802:	4b 8d 0c 3f          	lea    (%r15,%r15,1),%rcx
    2806:	0f b6 84 0d d1 fd ff 	movzbl -0x22f(%rbp,%rcx,1),%eax
    280d:	ff 
    280e:	88 47 01             	mov    %al,0x1(%rdi)
    2811:	0f b6 84 0d d0 fd ff 	movzbl -0x230(%rbp,%rcx,1),%eax
    2818:	ff 
    2819:	88 07                	mov    %al,(%rdi)
    281b:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    2822:	31 f6                	xor    %esi,%esi
    2824:	48 89 df             	mov    %rbx,%rdi
    2827:	4c 89 b5 b8 fd ff ff 	mov    %r14,-0x248(%rbp)
    282e:	48 8d 15 03 18 00 00 	lea    0x1803(%rip),%rdx        # 4038 <_IO_stdin_used+0x38>
    2835:	42 c6 04 30 00       	movb   $0x0,(%rax,%r14,1)
    283a:	c5 f8 77             	vzeroupper
    283d:	e8 1e f8 ff ff       	call   2060 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@plt>
    2842:	48 89 c6             	mov    %rax,%rsi
    2845:	4c 89 e7             	mov    %r12,%rdi
    2848:	e8 83 f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
    284d:	48 8d 35 f9 17 00 00 	lea    0x17f9(%rip),%rsi        # 404d <_IO_stdin_used+0x4d>
    2854:	4c 89 e7             	mov    %r12,%rdi
    2857:	e8 44 fa ff ff       	call   22a0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>
    285c:	48 89 c6             	mov    %rax,%rsi
    285f:	48 8d 85 90 fd ff ff 	lea    -0x270(%rbp),%rax
    2866:	4c 8d b5 c8 fe ff ff 	lea    -0x138(%rbp),%r14
    286d:	48 89 c7             	mov    %rax,%rdi
    2870:	48 89 85 20 fd ff ff 	mov    %rax,-0x2e0(%rbp)
    2877:	e8 54 f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
    287c:	4c 89 e7             	mov    %r12,%rdi
    287f:	e8 0c f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2884:	48 89 df             	mov    %rbx,%rdi
    2887:	e8 04 f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    288c:	4c 89 f7             	mov    %r14,%rdi
    288f:	4c 89 b5 10 fd ff ff 	mov    %r14,-0x2f0(%rbp)
    2896:	e8 e5 f7 ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
    289b:	48 8d 05 16 33 00 00 	lea    0x3316(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    28a2:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    28a6:	31 f6                	xor    %esi,%esi
    28a8:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    28af:	31 c0                	xor    %eax,%eax
    28b1:	66 89 45 a8          	mov    %ax,-0x58(%rbp)
    28b5:	48 8b 05 94 33 00 00 	mov    0x3394(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    28bc:	c5 fd 7f 45 b0       	vmovdqa %ymm0,-0x50(%rbp)
    28c1:	48 8b 78 e8          	mov    -0x18(%rax),%rdi
    28c5:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    28cc:	48 8b 05 85 33 00 00 	mov    0x3385(%rip),%rax        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    28d3:	48 c7 45 a0 00 00 00 	movq   $0x0,-0x60(%rbp)
    28da:	00 
    28db:	4c 01 e7             	add    %r12,%rdi
    28de:	48 89 07             	mov    %rax,(%rdi)
    28e1:	c5 f8 77             	vzeroupper
    28e4:	e8 c7 f8 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    28e9:	48 8d 05 60 34 00 00 	lea    0x3460(%rip),%rax        # 5d50 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    28f0:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    28f7:	48 83 c0 28          	add    $0x28,%rax
    28fb:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    2902:	48 8d 85 d8 fd ff ff 	lea    -0x228(%rbp),%rax
    2909:	48 89 c7             	mov    %rax,%rdi
    290c:	48 89 85 18 fd ff ff 	mov    %rax,-0x2e8(%rbp)
    2913:	48 89 c3             	mov    %rax,%rbx
    2916:	e8 45 f8 ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
    291b:	48 89 de             	mov    %rbx,%rsi
    291e:	4c 89 f7             	mov    %r14,%rdi
    2921:	e8 8a f8 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    2926:	48 8b b5 90 fd ff ff 	mov    -0x270(%rbp),%rsi
    292d:	ba 10 00 00 00       	mov    $0x10,%edx
    2932:	48 89 df             	mov    %rbx,%rdi
    2935:	e8 f6 f7 ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
    293a:	48 8b 95 d0 fd ff ff 	mov    -0x230(%rbp),%rdx
    2941:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    2945:	4c 01 e7             	add    %r12,%rdi
    2948:	48 85 c0             	test   %rax,%rax
    294b:	0f 84 0d 02 00 00    	je     2b5e <main+0x73e>
    2951:	31 f6                	xor    %esi,%esi
    2953:	e8 d8 f8 ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    2958:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    295f:	4c 8d 35 ae 16 00 00 	lea    0x16ae(%rip),%r14        # 4014 <_IO_stdin_used+0x14>
    2966:	4c 8d 2d e5 16 00 00 	lea    0x16e5(%rip),%r13        # 4052 <_IO_stdin_used+0x52>
    296d:	48 89 c3             	mov    %rax,%rbx
    2970:	48 39 85 48 fd ff ff 	cmp    %rax,-0x2b8(%rbp)
    2977:	74 58                	je     29d1 <main+0x5b1>
    2979:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2980:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    2984:	4c 89 e7             	mov    %r12,%rdi
    2987:	c5 f2 5a 03          	vcvtss2sd (%rbx),%xmm1,%xmm0
    298b:	e8 70 f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2990:	ba 01 00 00 00       	mov    $0x1,%edx
    2995:	4c 89 f6             	mov    %r14,%rsi
    2998:	48 89 c7             	mov    %rax,%rdi
    299b:	49 89 c7             	mov    %rax,%r15
    299e:	e8 cd f7 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    29a3:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    29a7:	4c 89 ff             	mov    %r15,%rdi
    29aa:	c5 f2 5a 43 04       	vcvtss2sd 0x4(%rbx),%xmm1,%xmm0
    29af:	e8 4c f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    29b4:	48 89 c7             	mov    %rax,%rdi
    29b7:	ba 01 00 00 00       	mov    $0x1,%edx
    29bc:	4c 89 ee             	mov    %r13,%rsi
    29bf:	e8 ac f7 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    29c4:	48 83 c3 08          	add    $0x8,%rbx
    29c8:	48 39 9d 48 fd ff ff 	cmp    %rbx,-0x2b8(%rbp)
    29cf:	75 af                	jne    2980 <main+0x560>
    29d1:	48 8d 05 a0 33 00 00 	lea    0x33a0(%rip),%rax        # 5d78 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x40>
    29d8:	c5 fa 7e 1d c8 33 00 	vmovq  0x33c8(%rip),%xmm3        # 5da8 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x70>
    29df:	00 
    29e0:	48 8b bd 18 fd ff ff 	mov    -0x2e8(%rbp),%rdi
    29e7:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    29ee:	48 8d 05 d3 32 00 00 	lea    0x32d3(%rip),%rax        # 5cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29f5:	c4 e3 e1 22 c0 01    	vpinsrq $0x1,%rax,%xmm3,%xmm0
    29fb:	c5 f9 7f 85 d0 fd ff 	vmovdqa %xmm0,-0x230(%rbp)
    2a02:	ff 
    2a03:	e8 48 f6 ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
    2a08:	48 8d bd 40 fe ff ff 	lea    -0x1c0(%rbp),%rdi
    2a0f:	e8 6c f8 ff ff       	call   2280 <_ZNSt12__basic_fileIcED1Ev@plt>
    2a14:	48 8d 05 bd 31 00 00 	lea    0x31bd(%rip),%rax        # 5bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2a1b:	48 8d bd 10 fe ff ff 	lea    -0x1f0(%rbp),%rdi
    2a22:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
    2a29:	e8 92 f7 ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
    2a2e:	48 8b 05 1b 32 00 00 	mov    0x321b(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    2a35:	48 8b 0d 1c 32 00 00 	mov    0x321c(%rip),%rcx        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2a3c:	48 8b bd 10 fd ff ff 	mov    -0x2f0(%rbp),%rdi
    2a43:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    2a4a:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2a4e:	48 89 8c 05 d0 fd ff 	mov    %rcx,-0x230(%rbp,%rax,1)
    2a55:	ff 
    2a56:	48 8d 05 5b 31 00 00 	lea    0x315b(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2a5d:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    2a64:	e8 27 f6 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    2a69:	48 8b bd 20 fd ff ff 	mov    -0x2e0(%rbp),%rdi
    2a70:	e8 1b f7 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2a75:	48 83 bd 40 fd ff ff 	cmpq   $0x0,-0x2c0(%rbp)
    2a7c:	00 
    2a7d:	74 13                	je     2a92 <main+0x672>
    2a7f:	48 8b b5 38 fd ff ff 	mov    -0x2c8(%rbp),%rsi
    2a86:	48 8b bd 40 fd ff ff 	mov    -0x2c0(%rbp),%rdi
    2a8d:	e8 be f6 ff ff       	call   2150 <_ZdlPvm@plt>
    2a92:	48 8b bd 30 fd ff ff 	mov    -0x2d0(%rbp),%rdi
    2a99:	48 85 ff             	test   %rdi,%rdi
    2a9c:	74 0f                	je     2aad <main+0x68d>
    2a9e:	48 8b b5 08 fd ff ff 	mov    -0x2f8(%rbp),%rsi
    2aa5:	48 29 fe             	sub    %rdi,%rsi
    2aa8:	e8 a3 f6 ff ff       	call   2150 <_ZdlPvm@plt>
    2aad:	31 c0                	xor    %eax,%eax
    2aaf:	48 81 c4 e0 02 00 00 	add    $0x2e0,%rsp
    2ab6:	5b                   	pop    %rbx
    2ab7:	41 5a                	pop    %r10
    2ab9:	41 5c                	pop    %r12
    2abb:	41 5d                	pop    %r13
    2abd:	41 5e                	pop    %r14
    2abf:	41 5f                	pop    %r15
    2ac1:	5d                   	pop    %rbp
    2ac2:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    2ac6:	c3                   	ret
    2ac7:	4c 89 e7             	mov    %r12,%rdi
    2aca:	48 8d b5 b0 fd ff ff 	lea    -0x250(%rbp),%rsi
    2ad1:	31 d2                	xor    %edx,%edx
    2ad3:	e8 98 f7 ff ff       	call   2270 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
    2ad8:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    2adf:	48 89 c7             	mov    %rax,%rdi
    2ae2:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    2ae9:	48 89 85 e0 fd ff ff 	mov    %rax,-0x220(%rbp)
    2af0:	48 89 da             	mov    %rbx,%rdx
    2af3:	4c 89 ee             	mov    %r13,%rsi
    2af6:	e8 15 f6 ff ff       	call   2110 <memcpy@plt>
    2afb:	e9 9f f9 ff ff       	jmp    249f <main+0x7f>
    2b00:	31 d2                	xor    %edx,%edx
    2b02:	48 89 95 28 fd ff ff 	mov    %rdx,-0x2d8(%rbp)
    2b09:	48 89 95 20 fd ff ff 	mov    %rdx,-0x2e0(%rbp)
    2b10:	48 89 95 40 fd ff ff 	mov    %rdx,-0x2c0(%rbp)
    2b17:	48 89 95 48 fd ff ff 	mov    %rdx,-0x2b8(%rbp)
    2b1e:	e9 e2 fa ff ff       	jmp    2605 <main+0x1e5>
    2b23:	48 8d 1d 76 37 00 00 	lea    0x3776(%rip),%rbx        # 62a0 <_ZSt4cerr@GLIBCXX_3.4>
    2b2a:	ba 1d 00 00 00       	mov    $0x1d,%edx
    2b2f:	48 8d 35 e4 14 00 00 	lea    0x14e4(%rip),%rsi        # 401a <_IO_stdin_used+0x1a>
    2b36:	48 89 df             	mov    %rbx,%rdi
    2b39:	e8 32 f6 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2b3e:	48 89 df             	mov    %rbx,%rdi
    2b41:	e8 ba 01 00 00       	call   2d00 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    2b46:	b8 01 00 00 00       	mov    $0x1,%eax
    2b4b:	e9 5f ff ff ff       	jmp    2aaf <main+0x68f>
    2b50:	48 85 c0             	test   %rax,%rax
    2b53:	0f 84 46 f9 ff ff    	je     249f <main+0x7f>
    2b59:	4c 89 f7             	mov    %r14,%rdi
    2b5c:	eb 92                	jmp    2af0 <main+0x6d0>
    2b5e:	8b 77 20             	mov    0x20(%rdi),%esi
    2b61:	83 ce 04             	or     $0x4,%esi
    2b64:	e8 c7 f6 ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    2b69:	e9 ea fd ff ff       	jmp    2958 <main+0x538>
    2b6e:	41 83 c5 02          	add    $0x2,%r13d
    2b72:	e9 66 fb ff ff       	jmp    26dd <main+0x2bd>
    2b77:	41 83 c5 03          	add    $0x3,%r13d
    2b7b:	e9 5d fb ff ff       	jmp    26dd <main+0x2bd>
    2b80:	c5 f8 77             	vzeroupper
    2b83:	e9 76 fa ff ff       	jmp    25fe <main+0x1de>
    2b88:	48 8b b5 40 fd ff ff 	mov    -0x2c0(%rbp),%rsi
    2b8f:	31 c0                	xor    %eax,%eax
    2b91:	e9 3a fa ff ff       	jmp    25d0 <main+0x1b0>
    2b96:	e9 5b f7 ff ff       	jmp    22f6 <main.cold+0xc>
    2b9b:	49 89 c4             	mov    %rax,%r12
    2b9e:	c5 f8 77             	vzeroupper
    2ba1:	e9 92 f7 ff ff       	jmp    2338 <main.cold+0x4e>
    2ba6:	48 89 c3             	mov    %rax,%rbx
    2ba9:	e9 a9 f7 ff ff       	jmp    2357 <main.cold+0x6d>
    2bae:	48 89 c7             	mov    %rax,%rdi
    2bb1:	e9 bd f7 ff ff       	jmp    2373 <main.cold+0x89>
    2bb6:	48 89 c3             	mov    %rax,%rbx
    2bb9:	e9 d3 f7 ff ff       	jmp    2391 <main.cold+0xa7>
    2bbe:	49 89 c4             	mov    %rax,%r12
    2bc1:	c5 f8 77             	vzeroupper
    2bc4:	e9 ea f7 ff ff       	jmp    23b3 <main.cold+0xc9>
    2bc9:	49 89 c4             	mov    %rax,%r12
    2bcc:	e9 d3 f7 ff ff       	jmp    23a4 <main.cold+0xba>
    2bd1:	49 89 c4             	mov    %rax,%r12
    2bd4:	e9 1a f8 ff ff       	jmp    23f3 <main.cold+0x109>
    2bd9:	49 89 c4             	mov    %rax,%r12
    2bdc:	c5 f8 77             	vzeroupper
    2bdf:	e9 4c f7 ff ff       	jmp    2330 <main.cold+0x46>
    2be4:	49 89 c5             	mov    %rax,%r13
    2be7:	e9 36 f7 ff ff       	jmp    2322 <main.cold+0x38>
    2bec:	49 89 c4             	mov    %rax,%r12
    2bef:	c5 f8 77             	vzeroupper
    2bf2:	e9 dd f7 ff ff       	jmp    23d4 <main.cold+0xea>
    2bf7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2bfe:	00 00 

0000000000002c00 <_start>:
    2c00:	31 ed                	xor    %ebp,%ebp
    2c02:	49 89 d1             	mov    %rdx,%r9
    2c05:	5e                   	pop    %rsi
    2c06:	48 89 e2             	mov    %rsp,%rdx
    2c09:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    2c0d:	50                   	push   %rax
    2c0e:	54                   	push   %rsp
    2c0f:	45 31 c0             	xor    %r8d,%r8d
    2c12:	31 c9                	xor    %ecx,%ecx
    2c14:	48 8d 3d 05 f8 ff ff 	lea    -0x7fb(%rip),%rdi        # 2420 <main>
    2c1b:	ff 15 a7 33 00 00    	call   *0x33a7(%rip)        # 5fc8 <__libc_start_main@GLIBC_2.34>
    2c21:	f4                   	hlt
    2c22:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2c29:	00 00 00 
    2c2c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002c30 <deregister_tm_clones>:
    2c30:	48 8d 3d 29 35 00 00 	lea    0x3529(%rip),%rdi        # 6160 <__TMC_END__>
    2c37:	48 8d 05 22 35 00 00 	lea    0x3522(%rip),%rax        # 6160 <__TMC_END__>
    2c3e:	48 39 f8             	cmp    %rdi,%rax
    2c41:	74 15                	je     2c58 <deregister_tm_clones+0x28>
    2c43:	48 8b 05 86 33 00 00 	mov    0x3386(%rip),%rax        # 5fd0 <_ITM_deregisterTMCloneTable@Base>
    2c4a:	48 85 c0             	test   %rax,%rax
    2c4d:	74 09                	je     2c58 <deregister_tm_clones+0x28>
    2c4f:	ff e0                	jmp    *%rax
    2c51:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2c58:	c3                   	ret
    2c59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002c60 <register_tm_clones>:
    2c60:	48 8d 3d f9 34 00 00 	lea    0x34f9(%rip),%rdi        # 6160 <__TMC_END__>
    2c67:	48 8d 35 f2 34 00 00 	lea    0x34f2(%rip),%rsi        # 6160 <__TMC_END__>
    2c6e:	48 29 fe             	sub    %rdi,%rsi
    2c71:	48 89 f0             	mov    %rsi,%rax
    2c74:	48 c1 ee 3f          	shr    $0x3f,%rsi
    2c78:	48 c1 f8 03          	sar    $0x3,%rax
    2c7c:	48 01 c6             	add    %rax,%rsi
    2c7f:	48 d1 fe             	sar    $1,%rsi
    2c82:	74 14                	je     2c98 <register_tm_clones+0x38>
    2c84:	48 8b 05 55 33 00 00 	mov    0x3355(%rip),%rax        # 5fe0 <_ITM_registerTMCloneTable@Base>
    2c8b:	48 85 c0             	test   %rax,%rax
    2c8e:	74 08                	je     2c98 <register_tm_clones+0x38>
    2c90:	ff e0                	jmp    *%rax
    2c92:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2c98:	c3                   	ret
    2c99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002ca0 <__do_global_dtors_aux>:
    2ca0:	f3 0f 1e fa          	endbr64
    2ca4:	80 3d 05 37 00 00 00 	cmpb   $0x0,0x3705(%rip)        # 63b0 <completed.0>
    2cab:	75 2b                	jne    2cd8 <__do_global_dtors_aux+0x38>
    2cad:	55                   	push   %rbp
    2cae:	48 83 3d 0a 33 00 00 	cmpq   $0x0,0x330a(%rip)        # 5fc0 <__cxa_finalize@GLIBC_2.2.5>
    2cb5:	00 
    2cb6:	48 89 e5             	mov    %rsp,%rbp
    2cb9:	74 0c                	je     2cc7 <__do_global_dtors_aux+0x27>
    2cbb:	48 8b 3d 8e 34 00 00 	mov    0x348e(%rip),%rdi        # 6150 <__dso_handle>
    2cc2:	e8 f9 f5 ff ff       	call   22c0 <__cxa_finalize@plt>
    2cc7:	e8 64 ff ff ff       	call   2c30 <deregister_tm_clones>
    2ccc:	c6 05 dd 36 00 00 01 	movb   $0x1,0x36dd(%rip)        # 63b0 <completed.0>
    2cd3:	5d                   	pop    %rbp
    2cd4:	c3                   	ret
    2cd5:	0f 1f 00             	nopl   (%rax)
    2cd8:	c3                   	ret
    2cd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002ce0 <frame_dummy>:
    2ce0:	f3 0f 1e fa          	endbr64
    2ce4:	e9 77 ff ff ff       	jmp    2c60 <register_tm_clones>
    2ce9:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2cf0:	00 00 00 
    2cf3:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2cfa:	00 00 00 
    2cfd:	0f 1f 00             	nopl   (%rax)

0000000000002d00 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>:
    2d00:	55                   	push   %rbp
    2d01:	53                   	push   %rbx
    2d02:	48 83 ec 08          	sub    $0x8,%rsp
    2d06:	48 8b 07             	mov    (%rdi),%rax
    2d09:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2d0d:	48 8b ac 07 f0 00 00 	mov    0xf0(%rdi,%rax,1),%rbp
    2d14:	00 
    2d15:	48 85 ed             	test   %rbp,%rbp
    2d18:	0f 84 c2 f5 ff ff    	je     22e0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0.cold>
    2d1e:	80 7d 38 00          	cmpb   $0x0,0x38(%rbp)
    2d22:	48 89 fb             	mov    %rdi,%rbx
    2d25:	74 1a                	je     2d41 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x41>
    2d27:	0f be 75 43          	movsbl 0x43(%rbp),%esi
    2d2b:	48 89 df             	mov    %rbx,%rdi
    2d2e:	e8 fd f2 ff ff       	call   2030 <_ZNSo3putEc@plt>
    2d33:	48 83 c4 08          	add    $0x8,%rsp
    2d37:	5b                   	pop    %rbx
    2d38:	48 89 c7             	mov    %rax,%rdi
    2d3b:	5d                   	pop    %rbp
    2d3c:	e9 9f f3 ff ff       	jmp    20e0 <_ZNSo5flushEv@plt>
    2d41:	48 89 ef             	mov    %rbp,%rdi
    2d44:	e8 37 f4 ff ff       	call   2180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
    2d49:	48 8b 45 00          	mov    0x0(%rbp),%rax
    2d4d:	be 0a 00 00 00       	mov    $0xa,%esi
    2d52:	48 8d 15 a7 09 00 00 	lea    0x9a7(%rip),%rdx        # 3700 <_ZNKSt5ctypeIcE8do_widenEc>
    2d59:	48 8b 40 30          	mov    0x30(%rax),%rax
    2d5d:	48 39 d0             	cmp    %rdx,%rax
    2d60:	74 c9                	je     2d2b <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x2b>
    2d62:	be 0a 00 00 00       	mov    $0xa,%esi
    2d67:	48 89 ef             	mov    %rbp,%rdi
    2d6a:	ff d0                	call   *%rax
    2d6c:	0f be f0             	movsbl %al,%esi
    2d6f:	eb ba                	jmp    2d2b <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x2b>
    2d71:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2d78:	00 00 00 00 
    2d7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002d80 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_>:
    2d80:	41 56                	push   %r14
    2d82:	41 55                	push   %r13
    2d84:	41 54                	push   %r12
    2d86:	55                   	push   %rbp
    2d87:	53                   	push   %rbx
    2d88:	48 83 ec 20          	sub    $0x20,%rsp
    2d8c:	48 85 ff             	test   %rdi,%rdi
    2d8f:	0f 84 03 01 00 00    	je     2e98 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x118>
    2d95:	48 8b 1e             	mov    (%rsi),%rbx
    2d98:	4c 8b 2a             	mov    (%rdx),%r13
    2d9b:	48 89 fd             	mov    %rdi,%rbp
    2d9e:	0f 88 01 01 00 00    	js     2ea5 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x125>
    2da4:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
    2da8:	c4 e1 c3 2a c7       	vcvtsi2sd %rdi,%xmm7,%xmm0
    2dad:	c5 f9 13 44 24 18    	vmovlpd %xmm0,0x18(%rsp)
    2db3:	45 31 e4             	xor    %r12d,%r12d
    2db6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2dbd:	00 00 00 
    2dc0:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    2dc4:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
    2dc8:	45 31 f6             	xor    %r14d,%r14d
    2dcb:	c4 c1 4b 2a c4       	vcvtsi2sd %r12d,%xmm6,%xmm0
    2dd0:	c5 fb 59 15 08 13 00 	vmulsd 0x1308(%rip),%xmm0,%xmm2        # 40e0 <_IO_stdin_used+0xe0>
    2dd7:	00 
    2dd8:	c5 f8 28 ec          	vmovaps %xmm4,%xmm5
    2ddc:	c5 fb 11 54 24 10    	vmovsd %xmm2,0x10(%rsp)
    2de2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2de8:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
    2dec:	c5 fa 11 6c 24 0c    	vmovss %xmm5,0xc(%rsp)
    2df2:	c4 c1 43 2a ce       	vcvtsi2sd %r14d,%xmm7,%xmm1
    2df7:	c5 f3 59 4c 24 10    	vmulsd 0x10(%rsp),%xmm1,%xmm1
    2dfd:	c5 fa 11 64 24 08    	vmovss %xmm4,0x8(%rsp)
    2e03:	c5 f3 5e 4c 24 18    	vdivsd 0x18(%rsp),%xmm1,%xmm1
    2e09:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
    2e0d:	c5 fa 11 4c 24 04    	vmovss %xmm1,0x4(%rsp)
    2e13:	c5 f0 57 05 d5 12 00 	vxorps 0x12d5(%rip),%xmm1,%xmm0        # 40f0 <_IO_stdin_used+0xf0>
    2e1a:	00 
    2e1b:	e8 00 f3 ff ff       	call   2120 <sinf@plt>
    2e20:	c5 fa 10 4c 24 04    	vmovss 0x4(%rsp),%xmm1
    2e26:	c5 fa 11 04 24       	vmovss %xmm0,(%rsp)
    2e2b:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    2e2f:	e8 cc f2 ff ff       	call   2100 <cosf@plt>
    2e34:	c5 fa 10 1c 24       	vmovss (%rsp),%xmm3
    2e39:	c4 a1 7a 10 4c f3 04 	vmovss 0x4(%rbx,%r14,8),%xmm1
    2e40:	c4 a1 7a 10 3c f3    	vmovss (%rbx,%r14,8),%xmm7
    2e46:	c5 fa 10 64 24 08    	vmovss 0x8(%rsp),%xmm4
    2e4c:	c5 f2 59 d0          	vmulss %xmm0,%xmm1,%xmm2
    2e50:	c5 fa 10 6c 24 0c    	vmovss 0xc(%rsp),%xmm5
    2e56:	c5 f2 59 f3          	vmulss %xmm3,%xmm1,%xmm6
    2e5a:	c4 e2 41 b9 d3       	vfmadd231ss %xmm3,%xmm7,%xmm2
    2e5f:	c4 e2 41 bb f0       	vfmsub231ss %xmm0,%xmm7,%xmm6
    2e64:	c5 f8 2e d6          	vucomiss %xmm6,%xmm2
    2e68:	7a 62                	jp     2ecc <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x14c>
    2e6a:	49 ff c6             	inc    %r14
    2e6d:	c5 d2 58 ee          	vaddss %xmm6,%xmm5,%xmm5
    2e71:	c5 da 58 e2          	vaddss %xmm2,%xmm4,%xmm4
    2e75:	4c 39 f5             	cmp    %r14,%rbp
    2e78:	0f 85 6a ff ff ff    	jne    2de8 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x68>
    2e7e:	c4 81 7a 11 6c e5 00 	vmovss %xmm5,0x0(%r13,%r12,8)
    2e85:	c4 81 7a 11 64 e5 04 	vmovss %xmm4,0x4(%r13,%r12,8)
    2e8c:	49 ff c4             	inc    %r12
    2e8f:	4c 39 e5             	cmp    %r12,%rbp
    2e92:	0f 85 28 ff ff ff    	jne    2dc0 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
    2e98:	48 83 c4 20          	add    $0x20,%rsp
    2e9c:	5b                   	pop    %rbx
    2e9d:	5d                   	pop    %rbp
    2e9e:	41 5c                	pop    %r12
    2ea0:	41 5d                	pop    %r13
    2ea2:	41 5e                	pop    %r14
    2ea4:	c3                   	ret
    2ea5:	48 89 f8             	mov    %rdi,%rax
    2ea8:	48 89 fa             	mov    %rdi,%rdx
    2eab:	c5 e1 57 db          	vxorpd %xmm3,%xmm3,%xmm3
    2eaf:	48 d1 e8             	shr    $1,%rax
    2eb2:	83 e2 01             	and    $0x1,%edx
    2eb5:	48 09 d0             	or     %rdx,%rax
    2eb8:	c4 e1 e3 2a c0       	vcvtsi2sd %rax,%xmm3,%xmm0
    2ebd:	c5 fb 58 d8          	vaddsd %xmm0,%xmm0,%xmm3
    2ec1:	c5 fb 11 5c 24 18    	vmovsd %xmm3,0x18(%rsp)
    2ec7:	e9 e7 fe ff ff       	jmp    2db3 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x33>
    2ecc:	c5 f8 28 d0          	vmovaps %xmm0,%xmm2
    2ed0:	c5 f8 28 c7          	vmovaps %xmm7,%xmm0
    2ed4:	c5 fa 11 6c 24 04    	vmovss %xmm5,0x4(%rsp)
    2eda:	49 ff c6             	inc    %r14
    2edd:	c5 fa 11 24 24       	vmovss %xmm4,(%rsp)
    2ee2:	e8 a9 f3 ff ff       	call   2290 <__mulsc3@plt>
    2ee7:	c5 fa 10 6c 24 04    	vmovss 0x4(%rsp),%xmm5
    2eed:	c5 fa 10 24 24       	vmovss (%rsp),%xmm4
    2ef2:	c4 e1 f9 7e c0       	vmovq  %xmm0,%rax
    2ef7:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    2efb:	48 c1 e8 20          	shr    $0x20,%rax
    2eff:	c5 d2 58 e9          	vaddss %xmm1,%xmm5,%xmm5
    2f03:	c5 f9 6e c0          	vmovd  %eax,%xmm0
    2f07:	c5 da 58 e0          	vaddss %xmm0,%xmm4,%xmm4
    2f0b:	4c 39 f5             	cmp    %r14,%rbp
    2f0e:	0f 85 d4 fe ff ff    	jne    2de8 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x68>
    2f14:	e9 65 ff ff ff       	jmp    2e7e <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0xfe>
    2f19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002f20 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_>:
    2f20:	41 57                	push   %r15
    2f22:	41 56                	push   %r14
    2f24:	41 55                	push   %r13
    2f26:	41 54                	push   %r12
    2f28:	55                   	push   %rbp
    2f29:	53                   	push   %rbx
    2f2a:	48 83 ec 68          	sub    $0x68,%rsp
    2f2e:	48 85 ff             	test   %rdi,%rdi
    2f31:	0f 84 c6 03 00 00    	je     32fd <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x3dd>
    2f37:	48 89 fd             	mov    %rdi,%rbp
    2f3a:	48 89 f3             	mov    %rsi,%rbx
    2f3d:	49 89 d7             	mov    %rdx,%r15
    2f40:	0f 88 cd 03 00 00    	js     3313 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x3f3>
    2f46:	c5 d1 57 ed          	vxorpd %xmm5,%xmm5,%xmm5
    2f4a:	c4 e1 d3 2a c7       	vcvtsi2sd %rdi,%xmm5,%xmm0
    2f4f:	c4 c1 f9 7e c4       	vmovq  %xmm0,%r12
    2f54:	45 31 ed             	xor    %r13d,%r13d
    2f57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2f5e:	00 00 
    2f60:	c4 c1 f9 6e c4       	vmovq  %r12,%xmm0
    2f65:	e8 f6 f2 ff ff       	call   2260 <log2@plt>
    2f6a:	c5 7b 2c c0          	vcvttsd2si %xmm0,%r8d
    2f6e:	45 85 c0             	test   %r8d,%r8d
    2f71:	0f 8e 95 03 00 00    	jle    330c <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x3ec>
    2f77:	4c 89 ee             	mov    %r13,%rsi
    2f7a:	31 c9                	xor    %ecx,%ecx
    2f7c:	31 c0                	xor    %eax,%eax
    2f7e:	66 90                	xchg   %ax,%ax
    2f80:	48 89 f7             	mov    %rsi,%rdi
    2f83:	48 01 c0             	add    %rax,%rax
    2f86:	ff c1                	inc    %ecx
    2f88:	48 d1 ee             	shr    $1,%rsi
    2f8b:	83 e7 01             	and    $0x1,%edi
    2f8e:	48 09 f8             	or     %rdi,%rax
    2f91:	41 39 c8             	cmp    %ecx,%r8d
    2f94:	75 ea                	jne    2f80 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x60>
    2f96:	48 c1 e0 03          	shl    $0x3,%rax
    2f9a:	48 8b 0b             	mov    (%rbx),%rcx
    2f9d:	49 8b 17             	mov    (%r15),%rdx
    2fa0:	c5 fa 10 04 01       	vmovss (%rcx,%rax,1),%xmm0
    2fa5:	c4 a1 7a 11 04 ea    	vmovss %xmm0,(%rdx,%r13,8)
    2fab:	c5 fa 10 44 01 04    	vmovss 0x4(%rcx,%rax,1),%xmm0
    2fb1:	c4 a1 7a 11 44 ea 04 	vmovss %xmm0,0x4(%rdx,%r13,8)
    2fb8:	49 ff c5             	inc    %r13
    2fbb:	4c 39 ed             	cmp    %r13,%rbp
    2fbe:	75 a0                	jne    2f60 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
    2fc0:	49 83 fd 01          	cmp    $0x1,%r13
    2fc4:	0f 84 33 03 00 00    	je     32fd <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x3dd>
    2fca:	c5 fa 10 2d 32 10 00 	vmovss 0x1032(%rip),%xmm5        # 4004 <_IO_stdin_used+0x4>
    2fd1:	00 
    2fd2:	4c 89 6c 24 38       	mov    %r13,0x38(%rsp)
    2fd7:	b9 02 00 00 00       	mov    $0x2,%ecx
    2fdc:	c7 44 24 2c 02 00 00 	movl   $0x2,0x2c(%rsp)
    2fe3:	00 
    2fe4:	c5 fa 11 6c 24 28    	vmovss %xmm5,0x28(%rsp)
    2fea:	c5 fa 10 2d 16 10 00 	vmovss 0x1016(%rip),%xmm5        # 4008 <_IO_stdin_used+0x8>
    2ff1:	00 
    2ff2:	c5 fa 11 6c 24 1c    	vmovss %xmm5,0x1c(%rsp)
    2ff8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2fff:	00 
    3000:	44 8b 6c 24 2c       	mov    0x2c(%rsp),%r13d
    3005:	48 89 c8             	mov    %rcx,%rax
    3008:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
    300d:	49 89 ce             	mov    %rcx,%r14
    3010:	48 f7 d8             	neg    %rax
    3013:	48 c1 e0 03          	shl    $0x3,%rax
    3017:	41 d1 fd             	sar    $1,%r13d
    301a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    301f:	49 63 c5             	movslq %r13d,%rax
    3022:	48 c1 e0 03          	shl    $0x3,%rax
    3026:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    302b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3030:	83 7c 24 2c 01       	cmpl   $0x1,0x2c(%rsp)
    3035:	0f 8e ba 01 00 00    	jle    31f5 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x2d5>
    303b:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    3040:	44 89 f3             	mov    %r14d,%ebx
    3043:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    3048:	45 31 e4             	xor    %r12d,%r12d
    304b:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
    3052:	00 
    3053:	c5 fa 10 7c 24 18    	vmovss 0x18(%rsp),%xmm7
    3059:	4a 8d 2c f0          	lea    (%rax,%r14,8),%rbp
    305d:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
    3061:	c7 44 24 08 00 00 80 	movl   $0x3f800000,0x8(%rsp)
    3068:	3f 
    3069:	c5 fa 10 74 24 08    	vmovss 0x8(%rsp),%xmm6
    306f:	29 c3                	sub    %eax,%ebx
    3071:	eb 73                	jmp    30e6 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1c6>
    3073:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3078:	0f be 77 43          	movsbl 0x43(%rdi),%esi
    307c:	4c 89 f7             	mov    %r14,%rdi
    307f:	e8 ac ef ff ff       	call   2030 <_ZNSo3putEc@plt>
    3084:	48 89 c7             	mov    %rax,%rdi
    3087:	e8 54 f0 ff ff       	call   20e0 <_ZNSo5flushEv@plt>
    308c:	c5 fa 10 74 24 08    	vmovss 0x8(%rsp),%xmm6
    3092:	c5 fa 10 7c 24 28    	vmovss 0x28(%rsp),%xmm7
    3098:	c5 fa 10 5c 24 18    	vmovss 0x18(%rsp),%xmm3
    309e:	c5 fa 10 64 24 1c    	vmovss 0x1c(%rsp),%xmm4
    30a4:	c5 ca 59 c7          	vmulss %xmm7,%xmm6,%xmm0
    30a8:	c5 c2 59 cb          	vmulss %xmm3,%xmm7,%xmm1
    30ac:	c4 e2 59 b9 c3       	vfmadd231ss %xmm3,%xmm4,%xmm0
    30b1:	c4 e2 49 bb cc       	vfmsub231ss %xmm4,%xmm6,%xmm1
    30b6:	c5 f8 2e c1          	vucomiss %xmm1,%xmm0
    30ba:	0f 8a 79 02 00 00    	jp     3339 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x419>
    30c0:	41 ff c4             	inc    %r12d
    30c3:	48 83 c5 08          	add    $0x8,%rbp
    30c7:	ff c3                	inc    %ebx
    30c9:	45 39 e5             	cmp    %r12d,%r13d
    30cc:	0f 8e 1e 01 00 00    	jle    31f0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x2d0>
    30d2:	c5 fa 11 44 24 18    	vmovss %xmm0,0x18(%rsp)
    30d8:	c5 f8 28 f1          	vmovaps %xmm1,%xmm6
    30dc:	c5 f8 28 f8          	vmovaps %xmm0,%xmm7
    30e0:	c5 fa 11 4c 24 08    	vmovss %xmm1,0x8(%rsp)
    30e6:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    30eb:	49 8b 07             	mov    (%r15),%rax
    30ee:	c5 fa 12 c6          	vmovsldup %xmm6,%xmm0
    30f2:	45 8d 74 1d 00       	lea    0x0(%r13,%rbx,1),%r14d
    30f7:	48 01 e9             	add    %rbp,%rcx
    30fa:	48 8d 14 28          	lea    (%rax,%rbp,1),%rdx
    30fe:	48 01 c8             	add    %rcx,%rax
    3101:	c5 fa 7e 22          	vmovq  (%rdx),%xmm4
    3105:	c5 fa 7e 28          	vmovq  (%rax),%xmm5
    3109:	c5 fa 7e c0          	vmovq  %xmm0,%xmm0
    310d:	c5 fa 7e cd          	vmovq  %xmm5,%xmm1
    3111:	c5 d0 c6 d5 e1       	vshufps $0xe1,%xmm5,%xmm5,%xmm2
    3116:	c5 f8 59 c1          	vmulps %xmm1,%xmm0,%xmm0
    311a:	c5 fa 12 cf          	vmovsldup %xmm7,%xmm1
    311e:	c5 fa 7e d2          	vmovq  %xmm2,%xmm2
    3122:	c5 fa 7e c9          	vmovq  %xmm1,%xmm1
    3126:	c5 e8 59 d1          	vmulps %xmm1,%xmm2,%xmm2
    312a:	c5 fa 7e c0          	vmovq  %xmm0,%xmm0
    312e:	c5 fa 7e d2          	vmovq  %xmm2,%xmm2
    3132:	c5 fb d0 c2          	vaddsubps %xmm2,%xmm0,%xmm0
    3136:	c5 fa 16 d0          	vmovshdup %xmm0,%xmm2
    313a:	c5 f8 28 c8          	vmovaps %xmm0,%xmm1
    313e:	c5 f8 2e d0          	vucomiss %xmm0,%xmm2
    3142:	0f 8a 18 02 00 00    	jp     3360 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x440>
    3148:	c5 fa 7e c9          	vmovq  %xmm1,%xmm1
    314c:	c5 fa 7e c4          	vmovq  %xmm4,%xmm0
    3150:	89 de                	mov    %ebx,%esi
    3152:	c5 f8 58 d1          	vaddps %xmm1,%xmm0,%xmm2
    3156:	c5 f8 5c c1          	vsubps %xmm1,%xmm0,%xmm0
    315a:	48 8d 3d 1f 30 00 00 	lea    0x301f(%rip),%rdi        # 6180 <_ZSt4cout@GLIBCXX_3.4>
    3161:	c5 f8 13 12          	vmovlps %xmm2,(%rdx)
    3165:	c5 f8 13 00          	vmovlps %xmm0,(%rax)
    3169:	e8 d2 f0 ff ff       	call   2240 <_ZNSolsEi@plt>
    316e:	ba 01 00 00 00       	mov    $0x1,%edx
    3173:	48 8d 35 9a 0e 00 00 	lea    0xe9a(%rip),%rsi        # 4014 <_IO_stdin_used+0x14>
    317a:	48 89 c7             	mov    %rax,%rdi
    317d:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    3182:	e8 e9 ef ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    3187:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    318c:	44 89 f6             	mov    %r14d,%esi
    318f:	e8 ac f0 ff ff       	call   2240 <_ZNSolsEi@plt>
    3194:	49 89 c6             	mov    %rax,%r14
    3197:	48 8b 00             	mov    (%rax),%rax
    319a:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    319e:	49 8b bc 06 f0 00 00 	mov    0xf0(%r14,%rax,1),%rdi
    31a5:	00 
    31a6:	48 85 ff             	test   %rdi,%rdi
    31a9:	0f 84 36 f1 ff ff    	je     22e5 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_.cold>
    31af:	80 7f 38 00          	cmpb   $0x0,0x38(%rdi)
    31b3:	0f 85 bf fe ff ff    	jne    3078 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x158>
    31b9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    31be:	e8 bd ef ff ff       	call   2180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
    31c3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    31c8:	be 0a 00 00 00       	mov    $0xa,%esi
    31cd:	48 8d 15 2c 05 00 00 	lea    0x52c(%rip),%rdx        # 3700 <_ZNKSt5ctypeIcE8do_widenEc>
    31d4:	48 8b 07             	mov    (%rdi),%rax
    31d7:	48 8b 40 30          	mov    0x30(%rax),%rax
    31db:	48 39 d0             	cmp    %rdx,%rax
    31de:	0f 84 98 fe ff ff    	je     307c <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x15c>
    31e4:	ff d0                	call   *%rax
    31e6:	0f be f0             	movsbl %al,%esi
    31e9:	e9 8e fe ff ff       	jmp    307c <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x15c>
    31ee:	66 90                	xchg   %ax,%ax
    31f0:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    31f5:	48 8d 1d 84 2f 00 00 	lea    0x2f84(%rip),%rbx        # 6180 <_ZSt4cout@GLIBCXX_3.4>
    31fc:	ba 03 00 00 00       	mov    $0x3,%edx
    3201:	48 8d 35 0e 0e 00 00 	lea    0xe0e(%rip),%rsi        # 4016 <_IO_stdin_used+0x16>
    3208:	48 8d 3d 71 2f 00 00 	lea    0x2f71(%rip),%rdi        # 6180 <_ZSt4cout@GLIBCXX_3.4>
    320f:	e8 5c ef ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    3214:	48 8b 03             	mov    (%rbx),%rax
    3217:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    321b:	48 8b 9c 03 f0 00 00 	mov    0xf0(%rbx,%rax,1),%rbx
    3222:	00 
    3223:	48 85 db             	test   %rbx,%rbx
    3226:	0f 84 b9 f0 ff ff    	je     22e5 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_.cold>
    322c:	80 7b 38 00          	cmpb   $0x0,0x38(%rbx)
    3230:	74 36                	je     3268 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x348>
    3232:	0f be 73 43          	movsbl 0x43(%rbx),%esi
    3236:	48 8d 3d 43 2f 00 00 	lea    0x2f43(%rip),%rdi        # 6180 <_ZSt4cout@GLIBCXX_3.4>
    323d:	e8 ee ed ff ff       	call   2030 <_ZNSo3putEc@plt>
    3242:	48 89 c7             	mov    %rax,%rdi
    3245:	e8 96 ee ff ff       	call   20e0 <_ZNSo5flushEv@plt>
    324a:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    324f:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    3254:	4c 01 f0             	add    %r14,%rax
    3257:	49 39 ce             	cmp    %rcx,%r14
    325a:	73 3c                	jae    3298 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x378>
    325c:	49 89 c6             	mov    %rax,%r14
    325f:	e9 cc fd ff ff       	jmp    3030 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x110>
    3264:	0f 1f 40 00          	nopl   0x0(%rax)
    3268:	48 89 df             	mov    %rbx,%rdi
    326b:	e8 10 ef ff ff       	call   2180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
    3270:	48 8b 03             	mov    (%rbx),%rax
    3273:	be 0a 00 00 00       	mov    $0xa,%esi
    3278:	48 8d 0d 81 04 00 00 	lea    0x481(%rip),%rcx        # 3700 <_ZNKSt5ctypeIcE8do_widenEc>
    327f:	48 8b 40 30          	mov    0x30(%rax),%rax
    3283:	48 39 c8             	cmp    %rcx,%rax
    3286:	74 ae                	je     3236 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x316>
    3288:	48 89 df             	mov    %rbx,%rdi
    328b:	ff d0                	call   *%rax
    328d:	0f be f0             	movsbl %al,%esi
    3290:	eb a4                	jmp    3236 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x316>
    3292:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    3298:	d1 64 24 2c          	shll   $1,0x2c(%rsp)
    329c:	48 63 4c 24 2c       	movslq 0x2c(%rsp),%rcx
    32a1:	48 39 4c 24 38       	cmp    %rcx,0x38(%rsp)
    32a6:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    32ab:	72 50                	jb     32fd <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x3dd>
    32ad:	c5 d1 57 ed          	vxorpd %xmm5,%xmm5,%xmm5
    32b1:	c5 fb 10 05 27 0e 00 	vmovsd 0xe27(%rip),%xmm0        # 40e0 <_IO_stdin_used+0xe0>
    32b8:	00 
    32b9:	c5 d3 2a c9          	vcvtsi2sd %ecx,%xmm5,%xmm1
    32bd:	c5 fb 5e c9          	vdivsd %xmm1,%xmm0,%xmm1
    32c1:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
    32c5:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    32c9:	c5 fa 11 4c 24 18    	vmovss %xmm1,0x18(%rsp)
    32cf:	e8 2c ee ff ff       	call   2100 <cosf@plt>
    32d4:	c5 fa 10 4c 24 18    	vmovss 0x18(%rsp),%xmm1
    32da:	c5 fa 11 44 24 1c    	vmovss %xmm0,0x1c(%rsp)
    32e0:	c5 f0 57 05 08 0e 00 	vxorps 0xe08(%rip),%xmm1,%xmm0        # 40f0 <_IO_stdin_used+0xf0>
    32e7:	00 
    32e8:	e8 33 ee ff ff       	call   2120 <sinf@plt>
    32ed:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    32f2:	c5 fa 11 44 24 28    	vmovss %xmm0,0x28(%rsp)
    32f8:	e9 03 fd ff ff       	jmp    3000 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0xe0>
    32fd:	48 83 c4 68          	add    $0x68,%rsp
    3301:	5b                   	pop    %rbx
    3302:	5d                   	pop    %rbp
    3303:	41 5c                	pop    %r12
    3305:	41 5d                	pop    %r13
    3307:	41 5e                	pop    %r14
    3309:	41 5f                	pop    %r15
    330b:	c3                   	ret
    330c:	31 c0                	xor    %eax,%eax
    330e:	e9 87 fc ff ff       	jmp    2f9a <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x7a>
    3313:	48 89 f8             	mov    %rdi,%rax
    3316:	48 89 fa             	mov    %rdi,%rdx
    3319:	c5 d1 57 ed          	vxorpd %xmm5,%xmm5,%xmm5
    331d:	48 d1 e8             	shr    $1,%rax
    3320:	83 e2 01             	and    $0x1,%edx
    3323:	48 09 d0             	or     %rdx,%rax
    3326:	c4 e1 d3 2a c0       	vcvtsi2sd %rax,%xmm5,%xmm0
    332b:	c5 fb 58 e8          	vaddsd %xmm0,%xmm0,%xmm5
    332f:	c4 c1 f9 7e ec       	vmovq  %xmm5,%r12
    3334:	e9 1b fc ff ff       	jmp    2f54 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x34>
    3339:	c5 f8 28 c4          	vmovaps %xmm4,%xmm0
    333d:	c5 f8 28 cf          	vmovaps %xmm7,%xmm1
    3341:	c5 f8 28 d6          	vmovaps %xmm6,%xmm2
    3345:	e8 46 ef ff ff       	call   2290 <__mulsc3@plt>
    334a:	c5 f9 6f e8          	vmovdqa %xmm0,%xmm5
    334e:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    3352:	c5 d0 c6 ed 55       	vshufps $0x55,%xmm5,%xmm5,%xmm5
    3357:	c5 f9 6f c5          	vmovdqa %xmm5,%xmm0
    335b:	e9 60 fd ff ff       	jmp    30c0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1a0>
    3360:	c5 fa 16 cd          	vmovshdup %xmm5,%xmm1
    3364:	c5 f8 28 c5          	vmovaps %xmm5,%xmm0
    3368:	c5 f8 28 df          	vmovaps %xmm7,%xmm3
    336c:	48 89 54 24 50       	mov    %rdx,0x50(%rsp)
    3371:	c5 f8 28 d6          	vmovaps %xmm6,%xmm2
    3375:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    337a:	c5 f8 13 64 24 58    	vmovlps %xmm4,0x58(%rsp)
    3380:	e8 0b ef ff ff       	call   2290 <__mulsc3@plt>
    3385:	c5 fa 7e 64 24 58    	vmovq  0x58(%rsp),%xmm4
    338b:	48 8b 54 24 50       	mov    0x50(%rsp),%rdx
    3390:	c4 e1 f9 7e c1       	vmovq  %xmm0,%rcx
    3395:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    339a:	48 c1 e9 20          	shr    $0x20,%rcx
    339e:	c5 f9 6e e9          	vmovd  %ecx,%xmm5
    33a2:	c5 f8 14 cd          	vunpcklps %xmm5,%xmm0,%xmm1
    33a6:	e9 9d fd ff ff       	jmp    3148 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x228>
    33ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000033b0 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_>:
    33b0:	41 57                	push   %r15
    33b2:	41 56                	push   %r14
    33b4:	41 55                	push   %r13
    33b6:	41 54                	push   %r12
    33b8:	55                   	push   %rbp
    33b9:	48 89 fd             	mov    %rdi,%rbp
    33bc:	53                   	push   %rbx
    33bd:	48 d1 ed             	shr    $1,%rbp
    33c0:	48 83 ec 78          	sub    $0x78,%rsp
    33c4:	48 8b 36             	mov    (%rsi),%rsi
    33c7:	4c 8b 22             	mov    (%rdx),%r12
    33ca:	85 ed                	test   %ebp,%ebp
    33cc:	0f 8e 93 02 00 00    	jle    3665 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x2b5>
    33d2:	4c 63 ed             	movslq %ebp,%r13
    33d5:	48 85 ff             	test   %rdi,%rdi
    33d8:	0f 88 ab 01 00 00    	js     3589 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1d9>
    33de:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
    33e2:	c4 e1 eb 2a f7       	vcvtsi2sd %rdi,%xmm2,%xmm6
    33e7:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    33ec:	4c 89 e3             	mov    %r12,%rbx
    33ef:	41 89 ee             	mov    %ebp,%r14d
    33f2:	4c 89 64 24 20       	mov    %r12,0x20(%rsp)
    33f7:	49 89 f4             	mov    %rsi,%r12
    33fa:	c5 fb 11 74 24 18    	vmovsd %xmm6,0x18(%rsp)
    3400:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
    3404:	43 8d 2c 36          	lea    (%r14,%r14,1),%ebp
    3408:	c5 eb 2a c5          	vcvtsi2sd %ebp,%xmm2,%xmm0
    340c:	c5 fb 59 0d d4 0c 00 	vmulsd 0xcd4(%rip),%xmm0,%xmm1        # 40e8 <_IO_stdin_used+0xe8>
    3413:	00 
    3414:	c5 f3 5e 4c 24 18    	vdivsd 0x18(%rsp),%xmm1,%xmm1
    341a:	c5 f3 5a c9          	vcvtsd2ss %xmm1,%xmm1,%xmm1
    341e:	c5 fa 11 4c 24 08    	vmovss %xmm1,0x8(%rsp)
    3424:	c5 f0 57 05 c4 0c 00 	vxorps 0xcc4(%rip),%xmm1,%xmm0        # 40f0 <_IO_stdin_used+0xf0>
    342b:	00 
    342c:	e8 ef ec ff ff       	call   2120 <sinf@plt>
    3431:	c5 fa 10 4c 24 08    	vmovss 0x8(%rsp),%xmm1
    3437:	c5 fa 11 44 24 0c    	vmovss %xmm0,0xc(%rsp)
    343d:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    3441:	e8 ba ec ff ff       	call   2100 <cosf@plt>
    3446:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    344b:	48 63 f5             	movslq %ebp,%rsi
    344e:	31 d2                	xor    %edx,%edx
    3450:	48 f7 f6             	div    %rsi
    3453:	41 89 c0             	mov    %eax,%r8d
    3456:	85 c0                	test   %eax,%eax
    3458:	0f 8e e2 00 00 00    	jle    3540 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x190>
    345e:	c5 fa 10 64 24 0c    	vmovss 0xc(%rsp),%xmm4
    3464:	4d 63 fe             	movslq %r14d,%r15
    3467:	48 c1 e6 03          	shl    $0x3,%rsi
    346b:	4c 89 e2             	mov    %r12,%rdx
    346e:	4a 8d 0c fd 00 00 00 	lea    0x0(,%r15,8),%rcx
    3475:	00 
    3476:	c5 fa 10 15 8e 0b 00 	vmovss 0xb8e(%rip),%xmm2        # 400c <_IO_stdin_used+0xc>
    347d:	00 
    347e:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    3482:	31 c0                	xor    %eax,%eax
    3484:	48 8d 2c 0b          	lea    (%rbx,%rcx,1),%rbp
    3488:	c5 78 28 c4          	vmovaps %xmm4,%xmm8
    348c:	c5 78 28 c8          	vmovaps %xmm0,%xmm9
    3490:	48 89 ef             	mov    %rbp,%rdi
    3493:	49 89 d1             	mov    %rdx,%r9
    3496:	48 29 cf             	sub    %rcx,%rdi
    3499:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    34a0:	c4 81 7a 10 4c f9 04 	vmovss 0x4(%r9,%r15,8),%xmm1
    34a7:	c4 81 7a 10 3c f9    	vmovss (%r9,%r15,8),%xmm7
    34ad:	c4 c1 7a 10 31       	vmovss (%r9),%xmm6
    34b2:	c4 c1 7a 10 69 04    	vmovss 0x4(%r9),%xmm5
    34b8:	c5 ea 59 c1          	vmulss %xmm1,%xmm2,%xmm0
    34bc:	c5 e2 59 e1          	vmulss %xmm1,%xmm3,%xmm4
    34c0:	c4 e2 61 b9 c7       	vfmadd231ss %xmm7,%xmm3,%xmm0
    34c5:	c4 e2 69 bb e7       	vfmsub231ss %xmm7,%xmm2,%xmm4
    34ca:	c5 f8 2e c4          	vucomiss %xmm4,%xmm0
    34ce:	0f 8a e5 00 00 00    	jp     35b9 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x209>
    34d4:	c5 da 58 ce          	vaddss %xmm6,%xmm4,%xmm1
    34d8:	c5 ca 5c e4          	vsubss %xmm4,%xmm6,%xmm4
    34dc:	49 83 c1 08          	add    $0x8,%r9
    34e0:	c5 fa 11 0f          	vmovss %xmm1,(%rdi)
    34e4:	c5 fa 58 cd          	vaddss %xmm5,%xmm0,%xmm1
    34e8:	c5 d2 5c c0          	vsubss %xmm0,%xmm5,%xmm0
    34ec:	c5 fa 11 4f 04       	vmovss %xmm1,0x4(%rdi)
    34f1:	c4 a1 7a 11 24 ef    	vmovss %xmm4,(%rdi,%r13,8)
    34f7:	c4 a1 7a 11 44 ef 04 	vmovss %xmm0,0x4(%rdi,%r13,8)
    34fe:	48 83 c7 08          	add    $0x8,%rdi
    3502:	48 39 fd             	cmp    %rdi,%rbp
    3505:	75 99                	jne    34a0 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0xf0>
    3507:	c5 ba 59 c2          	vmulss %xmm2,%xmm8,%xmm0
    350b:	c4 c1 62 59 c8       	vmulss %xmm8,%xmm3,%xmm1
    3510:	c4 c2 61 b9 c1       	vfmadd231ss %xmm9,%xmm3,%xmm0
    3515:	c4 e2 31 bb ca       	vfmsub231ss %xmm2,%xmm9,%xmm1
    351a:	c5 f8 2e c8          	vucomiss %xmm0,%xmm1
    351e:	0f 8a 49 01 00 00    	jp     366d <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x2bd>
    3524:	ff c0                	inc    %eax
    3526:	48 01 cd             	add    %rcx,%rbp
    3529:	48 01 f2             	add    %rsi,%rdx
    352c:	41 39 c0             	cmp    %eax,%r8d
    352f:	74 0f                	je     3540 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x190>
    3531:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
    3535:	c5 f8 28 d1          	vmovaps %xmm1,%xmm2
    3539:	e9 52 ff ff ff       	jmp    3490 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0xe0>
    353e:	66 90                	xchg   %ax,%ax
    3540:	41 d1 fe             	sar    $1,%r14d
    3543:	74 0e                	je     3553 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1a3>
    3545:	4c 89 e0             	mov    %r12,%rax
    3548:	49 89 dc             	mov    %rbx,%r12
    354b:	48 89 c3             	mov    %rax,%rbx
    354e:	e9 ad fe ff ff       	jmp    3400 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x50>
    3553:	4c 8b 64 24 20       	mov    0x20(%rsp),%r12
    3558:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    355d:	49 39 dc             	cmp    %rbx,%r12
    3560:	74 48                	je     35aa <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1fa>
    3562:	48 8d 14 fd 00 00 00 	lea    0x0(,%rdi,8),%rdx
    3569:	00 
    356a:	48 83 fa 08          	cmp    $0x8,%rdx
    356e:	7e 34                	jle    35a4 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1f4>
    3570:	48 83 c4 78          	add    $0x78,%rsp
    3574:	48 89 de             	mov    %rbx,%rsi
    3577:	4c 89 e7             	mov    %r12,%rdi
    357a:	5b                   	pop    %rbx
    357b:	5d                   	pop    %rbp
    357c:	41 5c                	pop    %r12
    357e:	41 5d                	pop    %r13
    3580:	41 5e                	pop    %r14
    3582:	41 5f                	pop    %r15
    3584:	e9 87 ec ff ff       	jmp    2210 <memmove@plt>
    3589:	48 89 f8             	mov    %rdi,%rax
    358c:	c5 e1 57 db          	vxorpd %xmm3,%xmm3,%xmm3
    3590:	83 e0 01             	and    $0x1,%eax
    3593:	48 09 e8             	or     %rbp,%rax
    3596:	c4 e1 e3 2a f0       	vcvtsi2sd %rax,%xmm3,%xmm6
    359b:	c5 cb 58 f6          	vaddsd %xmm6,%xmm6,%xmm6
    359f:	e9 43 fe ff ff       	jmp    33e7 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x37>
    35a4:	0f 84 2e 01 00 00    	je     36d8 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x328>
    35aa:	48 83 c4 78          	add    $0x78,%rsp
    35ae:	5b                   	pop    %rbx
    35af:	5d                   	pop    %rbp
    35b0:	41 5c                	pop    %r12
    35b2:	41 5d                	pop    %r13
    35b4:	41 5e                	pop    %r14
    35b6:	41 5f                	pop    %r15
    35b8:	c3                   	ret
    35b9:	c5 f8 28 c7          	vmovaps %xmm7,%xmm0
    35bd:	48 89 74 24 68       	mov    %rsi,0x68(%rsp)
    35c2:	4c 89 4c 24 60       	mov    %r9,0x60(%rsp)
    35c7:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
    35cc:	89 44 24 54          	mov    %eax,0x54(%rsp)
    35d0:	44 89 44 24 50       	mov    %r8d,0x50(%rsp)
    35d5:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
    35da:	48 89 7c 24 40       	mov    %rdi,0x40(%rsp)
    35df:	c5 7a 11 4c 24 38    	vmovss %xmm9,0x38(%rsp)
    35e5:	c5 7a 11 44 24 34    	vmovss %xmm8,0x34(%rsp)
    35eb:	c5 fa 11 74 24 30    	vmovss %xmm6,0x30(%rsp)
    35f1:	c5 fa 11 6c 24 28    	vmovss %xmm5,0x28(%rsp)
    35f7:	c5 fa 11 5c 24 0c    	vmovss %xmm3,0xc(%rsp)
    35fd:	c5 fa 11 54 24 08    	vmovss %xmm2,0x8(%rsp)
    3603:	e8 88 ec ff ff       	call   2290 <__mulsc3@plt>
    3608:	48 8b 74 24 68       	mov    0x68(%rsp),%rsi
    360d:	4c 8b 4c 24 60       	mov    0x60(%rsp),%r9
    3612:	c4 c1 f9 7e c2       	vmovq  %xmm0,%r10
    3617:	c5 f9 6f e0          	vmovdqa %xmm0,%xmm4
    361b:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
    3620:	8b 44 24 54          	mov    0x54(%rsp),%eax
    3624:	49 c1 ea 20          	shr    $0x20,%r10
    3628:	44 8b 44 24 50       	mov    0x50(%rsp),%r8d
    362d:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
    3632:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    3637:	c5 7a 10 4c 24 38    	vmovss 0x38(%rsp),%xmm9
    363d:	c4 c1 79 6e c2       	vmovd  %r10d,%xmm0
    3642:	c5 7a 10 44 24 34    	vmovss 0x34(%rsp),%xmm8
    3648:	c5 fa 10 74 24 30    	vmovss 0x30(%rsp),%xmm6
    364e:	c5 fa 10 6c 24 28    	vmovss 0x28(%rsp),%xmm5
    3654:	c5 fa 10 5c 24 0c    	vmovss 0xc(%rsp),%xmm3
    365a:	c5 fa 10 54 24 08    	vmovss 0x8(%rsp),%xmm2
    3660:	e9 6f fe ff ff       	jmp    34d4 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x124>
    3665:	48 89 f3             	mov    %rsi,%rbx
    3668:	e9 f0 fe ff ff       	jmp    355d <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1ad>
    366d:	c5 78 29 c8          	vmovaps %xmm9,%xmm0
    3671:	c5 78 29 c1          	vmovaps %xmm8,%xmm1
    3675:	48 89 74 24 40       	mov    %rsi,0x40(%rsp)
    367a:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
    367f:	89 44 24 34          	mov    %eax,0x34(%rsp)
    3683:	44 89 44 24 30       	mov    %r8d,0x30(%rsp)
    3688:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    368d:	c5 7a 11 44 24 0c    	vmovss %xmm8,0xc(%rsp)
    3693:	c5 7a 11 4c 24 08    	vmovss %xmm9,0x8(%rsp)
    3699:	e8 f2 eb ff ff       	call   2290 <__mulsc3@plt>
    369e:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    36a3:	48 8b 54 24 38       	mov    0x38(%rsp),%rdx
    36a8:	c4 e1 f9 7e c7       	vmovq  %xmm0,%rdi
    36ad:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    36b1:	8b 44 24 34          	mov    0x34(%rsp),%eax
    36b5:	44 8b 44 24 30       	mov    0x30(%rsp),%r8d
    36ba:	48 c1 ef 20          	shr    $0x20,%rdi
    36be:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    36c3:	c5 7a 10 44 24 0c    	vmovss 0xc(%rsp),%xmm8
    36c9:	c5 7a 10 4c 24 08    	vmovss 0x8(%rsp),%xmm9
    36cf:	c5 f9 6e c7          	vmovd  %edi,%xmm0
    36d3:	e9 4c fe ff ff       	jmp    3524 <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x174>
    36d8:	c5 fa 10 03          	vmovss (%rbx),%xmm0
    36dc:	c4 c1 7a 11 04 24    	vmovss %xmm0,(%r12)
    36e2:	c5 fa 10 43 04       	vmovss 0x4(%rbx),%xmm0
    36e7:	c4 c1 7a 11 44 24 04 	vmovss %xmm0,0x4(%r12)
    36ee:	e9 b7 fe ff ff       	jmp    35aa <_Z12fft_stockhammRSt6vectorISt7complexIfESaIS1_EES4_+0x1fa>
    36f3:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    36fa:	00 00 00 
    36fd:	0f 1f 00             	nopl   (%rax)

0000000000003700 <_ZNKSt5ctypeIcE8do_widenEc>:
    3700:	89 f0                	mov    %esi,%eax
    3702:	c3                   	ret
    3703:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    370a:	00 00 00 
    370d:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3714:	00 00 00 
    3717:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    371e:	00 00 

0000000000003720 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>:
    3720:	55                   	push   %rbp
    3721:	48 89 e5             	mov    %rsp,%rbp
    3724:	41 57                	push   %r15
    3726:	41 56                	push   %r14
    3728:	41 55                	push   %r13
    372a:	41 54                	push   %r12
    372c:	49 89 f4             	mov    %rsi,%r12
    372f:	53                   	push   %rbx
    3730:	48 89 fb             	mov    %rdi,%rbx
    3733:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    3737:	48 81 ec 60 02 00 00 	sub    $0x260,%rsp
    373e:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
    3745:	00 
    3746:	4c 8d 6c 24 40       	lea    0x40(%rsp),%r13
    374b:	48 89 c7             	mov    %rax,%rdi
    374e:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    3753:	e8 28 e9 ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
    3758:	4c 8b 3d 31 26 00 00 	mov    0x2631(%rip),%r15        # 5d90 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    375f:	31 d2                	xor    %edx,%edx
    3761:	31 f6                	xor    %esi,%esi
    3763:	48 8d 0d 4e 24 00 00 	lea    0x244e(%rip),%rcx        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    376a:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    376e:	66 89 94 24 20 02 00 	mov    %dx,0x220(%rsp)
    3775:	00 
    3776:	c5 fe 7f 84 24 28 02 	vmovdqu %ymm0,0x228(%rsp)
    377d:	00 00 
    377f:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    3783:	48 89 8c 24 40 01 00 	mov    %rcx,0x140(%rsp)
    378a:	00 
    378b:	48 8b 0d 06 26 00 00 	mov    0x2606(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3792:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
    3799:	00 00 00 00 00 
    379e:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    37a3:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    37a8:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    37af:	00 00 
    37b1:	49 8b 4f e8          	mov    -0x18(%r15),%rcx
    37b5:	4c 01 e9             	add    %r13,%rcx
    37b8:	48 89 cf             	mov    %rcx,%rdi
    37bb:	c5 f8 77             	vzeroupper
    37be:	e8 ed e9 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    37c3:	48 8d 0d b6 24 00 00 	lea    0x24b6(%rip),%rcx        # 5c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    37ca:	4c 8d 74 24 50       	lea    0x50(%rsp),%r14
    37cf:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
    37d4:	4c 89 f7             	mov    %r14,%rdi
    37d7:	48 83 c1 28          	add    $0x28,%rcx
    37db:	48 89 8c 24 40 01 00 	mov    %rcx,0x140(%rsp)
    37e2:	00 
    37e3:	4c 89 34 24          	mov    %r14,(%rsp)
    37e7:	e8 74 e9 ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
    37ec:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    37f1:	4c 89 f6             	mov    %r14,%rsi
    37f4:	e8 b7 e9 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    37f9:	49 8b 34 24          	mov    (%r12),%rsi
    37fd:	ba 08 00 00 00       	mov    $0x8,%edx
    3802:	4c 89 f7             	mov    %r14,%rdi
    3805:	e8 26 e9 ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
    380a:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    380f:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    3813:	4c 01 ef             	add    %r13,%rdi
    3816:	48 85 c0             	test   %rax,%rax
    3819:	0f 84 31 02 00 00    	je     3a50 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x330>
    381f:	31 f6                	xor    %esi,%esi
    3821:	e8 0a ea ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    3826:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
    382d:	00 
    382e:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    3832:	4c 8d 74 24 38       	lea    0x38(%rsp),%r14
    3837:	c5 fa 7f 03          	vmovdqu %xmm0,(%rbx)
    383b:	4c 89 f6             	mov    %r14,%rsi
    383e:	4c 89 ef             	mov    %r13,%rdi
    3841:	e8 6a ea ff ff       	call   22b0 <_ZNSi10_M_extractIfEERSiRT_@plt>
    3846:	48 89 c7             	mov    %rax,%rdi
    3849:	48 8d 74 24 3c       	lea    0x3c(%rsp),%rsi
    384e:	e8 5d ea ff ff       	call   22b0 <_ZNSi10_M_extractIfEERSiRT_@plt>
    3853:	48 8b 10             	mov    (%rax),%rdx
    3856:	48 8b 52 e8          	mov    -0x18(%rdx),%rdx
    385a:	f6 44 10 20 05       	testb  $0x5,0x20(%rax,%rdx,1)
    385f:	0f 85 4b 01 00 00    	jne    39b0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x290>
    3865:	48 8b 53 08          	mov    0x8(%rbx),%rdx
    3869:	48 3b 53 10          	cmp    0x10(%rbx),%rdx
    386d:	74 31                	je     38a0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x180>
    386f:	c5 fa 10 44 24 38    	vmovss 0x38(%rsp),%xmm0
    3875:	48 83 c2 08          	add    $0x8,%rdx
    3879:	4c 89 f6             	mov    %r14,%rsi
    387c:	4c 89 ef             	mov    %r13,%rdi
    387f:	c4 e3 79 21 44 24 3c 	vinsertps $0x10,0x3c(%rsp),%xmm0,%xmm0
    3886:	10 
    3887:	c5 f8 13 42 f8       	vmovlps %xmm0,-0x8(%rdx)
    388c:	48 89 53 08          	mov    %rdx,0x8(%rbx)
    3890:	e8 1b ea ff ff       	call   22b0 <_ZNSi10_M_extractIfEERSiRT_@plt>
    3895:	eb af                	jmp    3846 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x126>
    3897:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    389e:	00 00 
    38a0:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    38a7:	ff ff 0f 
    38aa:	48 8b 0b             	mov    (%rbx),%rcx
    38ad:	48 89 d6             	mov    %rdx,%rsi
    38b0:	48 29 ce             	sub    %rcx,%rsi
    38b3:	49 89 f4             	mov    %rsi,%r12
    38b6:	49 c1 fc 03          	sar    $0x3,%r12
    38ba:	49 39 c4             	cmp    %rax,%r12
    38bd:	0f 84 1c 02 00 00    	je     3adf <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3bf>
    38c3:	4d 85 e4             	test   %r12,%r12
    38c6:	b8 01 00 00 00       	mov    $0x1,%eax
    38cb:	49 0f 45 c4          	cmovne %r12,%rax
    38cf:	49 01 c4             	add    %rax,%r12
    38d2:	0f 82 88 01 00 00    	jb     3a60 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x340>
    38d8:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    38df:	ff ff 0f 
    38e2:	49 39 c4             	cmp    %rax,%r12
    38e5:	4c 0f 47 e0          	cmova  %rax,%r12
    38e9:	49 c1 e4 03          	shl    $0x3,%r12
    38ed:	4c 89 e7             	mov    %r12,%rdi
    38f0:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
    38f5:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    38fa:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
    38ff:	e8 3c e8 ff ff       	call   2140 <_Znwm@plt>
    3904:	c5 fa 10 44 24 38    	vmovss 0x38(%rsp),%xmm0
    390a:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    390f:	49 89 c0             	mov    %rax,%r8
    3912:	c4 e3 79 21 44 24 3c 	vinsertps $0x10,0x3c(%rsp),%xmm0,%xmm0
    3919:	10 
    391a:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    391f:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    3924:	c5 f8 13 04 30       	vmovlps %xmm0,(%rax,%rsi,1)
    3929:	48 39 ca             	cmp    %rcx,%rdx
    392c:	74 32                	je     3960 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x240>
    392e:	48 29 ca             	sub    %rcx,%rdx
    3931:	48 89 ce             	mov    %rcx,%rsi
    3934:	48 01 c2             	add    %rax,%rdx
    3937:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    393e:	00 00 
    3940:	c5 fa 10 06          	vmovss (%rsi),%xmm0
    3944:	48 83 c0 08          	add    $0x8,%rax
    3948:	48 83 c6 08          	add    $0x8,%rsi
    394c:	c5 fa 11 40 f8       	vmovss %xmm0,-0x8(%rax)
    3951:	c5 fa 10 46 fc       	vmovss -0x4(%rsi),%xmm0
    3956:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
    395b:	48 39 d0             	cmp    %rdx,%rax
    395e:	75 e0                	jne    3940 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x220>
    3960:	48 83 c0 08          	add    $0x8,%rax
    3964:	c4 c1 f9 6e c8       	vmovq  %r8,%xmm1
    3969:	c4 e3 f1 22 c0 01    	vpinsrq $0x1,%rax,%xmm1,%xmm0
    396f:	48 85 c9             	test   %rcx,%rcx
    3972:	74 25                	je     3999 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x279>
    3974:	48 8b 73 10          	mov    0x10(%rbx),%rsi
    3978:	48 89 cf             	mov    %rcx,%rdi
    397b:	4c 89 44 24 20       	mov    %r8,0x20(%rsp)
    3980:	c5 f9 7f 44 24 10    	vmovdqa %xmm0,0x10(%rsp)
    3986:	48 29 ce             	sub    %rcx,%rsi
    3989:	e8 c2 e7 ff ff       	call   2150 <_ZdlPvm@plt>
    398e:	4c 8b 44 24 20       	mov    0x20(%rsp),%r8
    3993:	c5 f9 6f 44 24 10    	vmovdqa 0x10(%rsp),%xmm0
    3999:	4d 01 e0             	add    %r12,%r8
    399c:	c5 fa 7f 03          	vmovdqu %xmm0,(%rbx)
    39a0:	4c 89 43 10          	mov    %r8,0x10(%rbx)
    39a4:	e9 92 fe ff ff       	jmp    383b <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x11b>
    39a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    39b0:	48 8d 05 c9 22 00 00 	lea    0x22c9(%rip),%rax        # 5c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    39b7:	48 8b 3c 24          	mov    (%rsp),%rdi
    39bb:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    39c0:	48 83 c0 28          	add    $0x28,%rax
    39c4:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    39cb:	00 
    39cc:	48 8d 05 f5 22 00 00 	lea    0x22f5(%rip),%rax        # 5cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    39d3:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    39d8:	e8 73 e6 ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
    39dd:	48 8d bc 24 b8 00 00 	lea    0xb8(%rsp),%rdi
    39e4:	00 
    39e5:	e8 96 e8 ff ff       	call   2280 <_ZNSt12__basic_fileIcED1Ev@plt>
    39ea:	48 8d 05 e7 21 00 00 	lea    0x21e7(%rip),%rax        # 5bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    39f1:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
    39f8:	00 
    39f9:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    39fe:	e8 bd e7 ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
    3a03:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    3a07:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    3a0c:	48 8b 0d 85 23 00 00 	mov    0x2385(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3a13:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    3a18:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    3a1d:	48 8d 05 94 21 00 00 	lea    0x2194(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3a24:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    3a2b:	00 
    3a2c:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    3a33:	00 00 
    3a35:	e8 56 e6 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    3a3a:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    3a3e:	48 89 d8             	mov    %rbx,%rax
    3a41:	5b                   	pop    %rbx
    3a42:	41 5c                	pop    %r12
    3a44:	41 5d                	pop    %r13
    3a46:	41 5e                	pop    %r14
    3a48:	41 5f                	pop    %r15
    3a4a:	5d                   	pop    %rbp
    3a4b:	c3                   	ret
    3a4c:	0f 1f 40 00          	nopl   0x0(%rax)
    3a50:	8b 77 20             	mov    0x20(%rdi),%esi
    3a53:	83 ce 04             	or     $0x4,%esi
    3a56:	e8 d5 e7 ff ff       	call   2230 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    3a5b:	e9 c6 fd ff ff       	jmp    3826 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x106>
    3a60:	49 bc f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%r12
    3a67:	ff ff 7f 
    3a6a:	e9 7e fe ff ff       	jmp    38ed <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1cd>
    3a6f:	48 89 c3             	mov    %rax,%rbx
    3a72:	eb 11                	jmp    3a85 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x365>
    3a74:	48 89 c3             	mov    %rax,%rbx
    3a77:	eb 28                	jmp    3aa1 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x381>
    3a79:	48 8b 3c 24          	mov    (%rsp),%rdi
    3a7d:	c5 f8 77             	vzeroupper
    3a80:	e8 5b e7 ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
    3a85:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    3a89:	48 8b 0d 08 23 00 00 	mov    0x2308(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3a90:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    3a95:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    3a9a:	31 c0                	xor    %eax,%eax
    3a9c:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    3aa1:	48 8d 05 10 21 00 00 	lea    0x2110(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3aa8:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    3aad:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    3ab4:	00 
    3ab5:	c5 f8 77             	vzeroupper
    3ab8:	e8 d3 e5 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    3abd:	48 89 df             	mov    %rbx,%rdi
    3ac0:	e8 8b e7 ff ff       	call   2250 <_Unwind_Resume@plt>
    3ac5:	48 89 c3             	mov    %rax,%rbx
    3ac8:	eb af                	jmp    3a79 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x359>
    3aca:	48 89 c7             	mov    %rax,%rdi
    3acd:	c5 f8 77             	vzeroupper
    3ad0:	e8 cb e5 ff ff       	call   20a0 <__cxa_begin_catch@plt>
    3ad5:	e8 46 e7 ff ff       	call   2220 <__cxa_end_catch@plt>
    3ada:	e9 fe fe ff ff       	jmp    39dd <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2bd>
    3adf:	48 8d 3d 6e 05 00 00 	lea    0x56e(%rip),%rdi        # 4054 <_IO_stdin_used+0x54>
    3ae6:	e8 d5 e5 ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
    3aeb:	49 89 c4             	mov    %rax,%r12
    3aee:	48 8b 3b             	mov    (%rbx),%rdi
    3af1:	48 8b 73 10          	mov    0x10(%rbx),%rsi
    3af5:	48 29 fe             	sub    %rdi,%rsi
    3af8:	48 85 ff             	test   %rdi,%rdi
    3afb:	74 18                	je     3b15 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3f5>
    3afd:	c5 f8 77             	vzeroupper
    3b00:	e8 4b e6 ff ff       	call   2150 <_ZdlPvm@plt>
    3b05:	4c 89 ef             	mov    %r13,%rdi
    3b08:	e8 63 e5 ff ff       	call   2070 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@plt>
    3b0d:	4c 89 e7             	mov    %r12,%rdi
    3b10:	e8 3b e7 ff ff       	call   2250 <_Unwind_Resume@plt>
    3b15:	c5 f8 77             	vzeroupper
    3b18:	eb eb                	jmp    3b05 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3e5>

Disassembly of section .fini:

0000000000003b1c <_fini>:
    3b1c:	48 83 ec 08          	sub    $0x8,%rsp
    3b20:	48 83 c4 08          	add    $0x8,%rsp
    3b24:	c3                   	ret
