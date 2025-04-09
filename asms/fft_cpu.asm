
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

0000000000002210 <__cxa_end_catch@plt>:
    2210:	ff 25 da 3e 00 00    	jmp    *0x3eda(%rip)        # 60f0 <__cxa_end_catch@CXXABI_1.3>
    2216:	68 1e 00 00 00       	push   $0x1e
    221b:	e9 00 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>:
    2220:	ff 25 d2 3e 00 00    	jmp    *0x3ed2(%rip)        # 60f8 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@GLIBCXX_3.4>
    2226:	68 1f 00 00 00       	push   $0x1f
    222b:	e9 f0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002230 <_Unwind_Resume@plt>:
    2230:	ff 25 ca 3e 00 00    	jmp    *0x3eca(%rip)        # 6100 <_Unwind_Resume@GCC_3.0>
    2236:	68 20 00 00 00       	push   $0x20
    223b:	e9 e0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002240 <log2@plt>:
    2240:	ff 25 c2 3e 00 00    	jmp    *0x3ec2(%rip)        # 6108 <log2@GLIBC_2.29>
    2246:	68 21 00 00 00       	push   $0x21
    224b:	e9 d0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002250 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>:
    2250:	ff 25 ba 3e 00 00    	jmp    *0x3eba(%rip)        # 6110 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@GLIBCXX_3.4.21>
    2256:	68 22 00 00 00       	push   $0x22
    225b:	e9 c0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002260 <_ZNSt12__basic_fileIcED1Ev@plt>:
    2260:	ff 25 b2 3e 00 00    	jmp    *0x3eb2(%rip)        # 6118 <_ZNSt12__basic_fileIcED1Ev@GLIBCXX_3.4>
    2266:	68 23 00 00 00       	push   $0x23
    226b:	e9 b0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002270 <__mulsc3@plt>:
    2270:	ff 25 aa 3e 00 00    	jmp    *0x3eaa(%rip)        # 6120 <__mulsc3@GCC_4.0.0>
    2276:	68 24 00 00 00       	push   $0x24
    227b:	e9 a0 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002280 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>:
    2280:	ff 25 a2 3e 00 00    	jmp    *0x3ea2(%rip)        # 6128 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@GLIBCXX_3.4.21>
    2286:	68 25 00 00 00       	push   $0x25
    228b:	e9 90 fd ff ff       	jmp    2020 <_init+0x20>

0000000000002290 <_ZNSi10_M_extractIfEERSiRT_@plt>:
    2290:	ff 25 9a 3e 00 00    	jmp    *0x3e9a(%rip)        # 6130 <_ZNSi10_M_extractIfEERSiRT_@GLIBCXX_3.4.9>
    2296:	68 26 00 00 00       	push   $0x26
    229b:	e9 80 fd ff ff       	jmp    2020 <_init+0x20>

Disassembly of section .plt.got:

00000000000022a0 <__cxa_finalize@plt>:
    22a0:	ff 25 1a 3d 00 00    	jmp    *0x3d1a(%rip)        # 5fc0 <__cxa_finalize@GLIBC_2.2.5>
    22a6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000000022c0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0.cold>:
    22c0:	e8 db fe ff ff       	call   21a0 <_ZSt16__throw_bad_castv@plt>

00000000000022c5 <main.cold>:
    22c5:	48 8d 3d a4 1d 00 00 	lea    0x1da4(%rip),%rdi        # 4070 <_IO_stdin_used+0x70>
    22cc:	e8 1f fe ff ff       	call   20f0 <_ZSt19__throw_logic_errorPKc@plt>
    22d1:	49 89 c4             	mov    %rax,%r12
    22d4:	c5 f8 77             	vzeroupper
    22d7:	48 8b b5 08 fd ff ff 	mov    -0x2f8(%rbp),%rsi
    22de:	48 8b 85 30 fd ff ff 	mov    -0x2d0(%rbp),%rax
    22e5:	48 29 c6             	sub    %rax,%rsi
    22e8:	48 85 c0             	test   %rax,%rax
    22eb:	74 08                	je     22f5 <main.cold+0x30>
    22ed:	48 89 c7             	mov    %rax,%rdi
    22f0:	e8 5b fe ff ff       	call   2150 <_ZdlPvm@plt>
    22f5:	4c 89 e7             	mov    %r12,%rdi
    22f8:	e8 33 ff ff ff       	call   2230 <_Unwind_Resume@plt>
    22fd:	4c 89 e7             	mov    %r12,%rdi
    2300:	c5 f8 77             	vzeroupper
    2303:	4d 89 ec             	mov    %r13,%r12
    2306:	e8 85 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    230b:	48 89 df             	mov    %rbx,%rdi
    230e:	e8 7d fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2313:	48 83 bd 40 fd ff ff 	cmpq   $0x0,-0x2c0(%rbp)
    231a:	00 
    231b:	74 ba                	je     22d7 <main.cold+0x12>
    231d:	48 8b b5 28 fd ff ff 	mov    -0x2d8(%rbp),%rsi
    2324:	48 8b bd 40 fd ff ff 	mov    -0x2c0(%rbp),%rdi
    232b:	e8 20 fe ff ff       	call   2150 <_ZdlPvm@plt>
    2330:	eb a5                	jmp    22d7 <main.cold+0x12>
    2332:	4c 89 e7             	mov    %r12,%rdi
    2335:	c5 f8 77             	vzeroupper
    2338:	49 89 dc             	mov    %rbx,%r12
    233b:	e8 90 fe ff ff       	call   21d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    2340:	48 8b bd 20 fd ff ff 	mov    -0x2e0(%rbp),%rdi
    2347:	e8 44 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    234c:	eb c5                	jmp    2313 <main.cold+0x4e>
    234e:	c5 f8 77             	vzeroupper
    2351:	e8 4a fd ff ff       	call   20a0 <__cxa_begin_catch@plt>
    2356:	e8 b5 fe ff ff       	call   2210 <__cxa_end_catch@plt>
    235b:	e9 68 06 00 00       	jmp    29c8 <main+0x5e8>
    2360:	48 8d 3d 41 1d 00 00 	lea    0x1d41(%rip),%rdi        # 40a8 <_IO_stdin_used+0xa8>
    2367:	e8 54 fd ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
    236c:	4c 89 e7             	mov    %r12,%rdi
    236f:	c5 f8 77             	vzeroupper
    2372:	e8 19 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2377:	48 89 df             	mov    %rbx,%rdi
    237a:	e8 b1 fe ff ff       	call   2230 <_Unwind_Resume@plt>
    237f:	48 8b bd 18 fd ff ff 	mov    -0x2e8(%rbp),%rdi
    2386:	c5 f8 77             	vzeroupper
    2389:	e8 52 fe ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
    238e:	48 8b 05 bb 38 00 00 	mov    0x38bb(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    2395:	48 8b 0d bc 38 00 00 	mov    0x38bc(%rip),%rcx        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    239c:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    23a3:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    23a7:	48 89 8c 05 d0 fd ff 	mov    %rcx,-0x230(%rbp,%rax,1)
    23ae:	ff 
    23af:	48 8b bd 10 fd ff ff 	mov    -0x2f0(%rbp),%rdi
    23b6:	48 8d 05 fb 37 00 00 	lea    0x37fb(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    23bd:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    23c4:	e8 c7 fc ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    23c9:	e9 72 ff ff ff       	jmp    2340 <main.cold+0x7b>
    23ce:	48 89 df             	mov    %rbx,%rdi
    23d1:	c5 f8 77             	vzeroupper
    23d4:	e8 b7 fd ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    23d9:	e9 35 ff ff ff       	jmp    2313 <main.cold+0x4e>
    23de:	66 90                	xchg   %ax,%ax

00000000000023e0 <main>:
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
    2403:	83 ff 01             	cmp    $0x1,%edi
    2406:	0f 8e d7 06 00 00    	jle    2ae3 <main+0x703>
    240c:	4c 8b 6e 08          	mov    0x8(%rsi),%r13
    2410:	4c 8d b5 e0 fd ff ff 	lea    -0x220(%rbp),%r14
    2417:	4c 8d a5 d0 fd ff ff 	lea    -0x230(%rbp),%r12
    241e:	4c 89 b5 d0 fd ff ff 	mov    %r14,-0x230(%rbp)
    2425:	4d 85 ed             	test   %r13,%r13
    2428:	0f 84 97 fe ff ff    	je     22c5 <main.cold>
    242e:	4c 89 ef             	mov    %r13,%rdi
    2431:	e8 7a fc ff ff       	call   20b0 <strlen@plt>
    2436:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
    243d:	48 89 c3             	mov    %rax,%rbx
    2440:	48 83 f8 0f          	cmp    $0xf,%rax
    2444:	0f 87 3d 06 00 00    	ja     2a87 <main+0x6a7>
    244a:	48 83 f8 01          	cmp    $0x1,%rax
    244e:	0f 85 bc 06 00 00    	jne    2b10 <main+0x730>
    2454:	41 0f b6 45 00       	movzbl 0x0(%r13),%eax
    2459:	88 85 e0 fd ff ff    	mov    %al,-0x220(%rbp)
    245f:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    2466:	48 8b 95 d0 fd ff ff 	mov    -0x230(%rbp),%rdx
    246d:	4c 8d b5 50 fd ff ff 	lea    -0x2b0(%rbp),%r14
    2474:	4c 89 e6             	mov    %r12,%rsi
    2477:	4c 89 f7             	mov    %r14,%rdi
    247a:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
    2481:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
    2485:	e8 16 0e 00 00       	call   32a0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>
    248a:	48 8b 85 60 fd ff ff 	mov    -0x2a0(%rbp),%rax
    2491:	4c 89 e7             	mov    %r12,%rdi
    2494:	48 89 85 08 fd ff ff 	mov    %rax,-0x2f8(%rbp)
    249b:	e8 f0 fc ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    24a0:	48 8b 8d 50 fd ff ff 	mov    -0x2b0(%rbp),%rcx
    24a7:	48 8b 85 58 fd ff ff 	mov    -0x2a8(%rbp),%rax
    24ae:	48 29 c8             	sub    %rcx,%rax
    24b1:	48 89 8d 30 fd ff ff 	mov    %rcx,-0x2d0(%rbp)
    24b8:	48 89 c3             	mov    %rax,%rbx
    24bb:	48 89 85 28 fd ff ff 	mov    %rax,-0x2d8(%rbp)
    24c2:	48 89 c1             	mov    %rax,%rcx
    24c5:	48 89 85 38 fd ff ff 	mov    %rax,-0x2c8(%rbp)
    24cc:	48 c1 fb 03          	sar    $0x3,%rbx
    24d0:	48 b8 f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%rax
    24d7:	ff ff 7f 
    24da:	49 89 df             	mov    %rbx,%r15
    24dd:	48 39 c8             	cmp    %rcx,%rax
    24e0:	0f 82 7a fe ff ff    	jb     2360 <main.cold+0x9b>
    24e6:	48 c7 85 78 fd ff ff 	movq   $0x0,-0x288(%rbp)
    24ed:	00 00 00 00 
    24f1:	48 85 db             	test   %rbx,%rbx
    24f4:	0f 84 c6 05 00 00    	je     2ac0 <main+0x6e0>
    24fa:	4c 8b ad 28 fd ff ff 	mov    -0x2d8(%rbp),%r13
    2501:	4c 89 ef             	mov    %r13,%rdi
    2504:	e8 37 fc ff ff       	call   2140 <_Znwm@plt>
    2509:	4a 8d 3c 28          	lea    (%rax,%r13,1),%rdi
    250d:	48 89 85 40 fd ff ff 	mov    %rax,-0x2c0(%rbp)
    2514:	48 89 c1             	mov    %rax,%rcx
    2517:	48 8d 43 ff          	lea    -0x1(%rbx),%rax
    251b:	48 89 bd 20 fd ff ff 	mov    %rdi,-0x2e0(%rbp)
    2522:	48 83 f8 02          	cmp    $0x2,%rax
    2526:	0f 86 1c 06 00 00    	jbe    2b48 <main+0x768>
    252c:	48 89 de             	mov    %rbx,%rsi
    252f:	48 89 c8             	mov    %rcx,%rax
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
    2564:	48 83 c0 40          	add    $0x40,%rax
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
    25c5:	48 8b 85 48 fd ff ff 	mov    -0x2b8(%rbp),%rax
    25cc:	48 89 85 78 fd ff ff 	mov    %rax,-0x288(%rbp)
    25d3:	e8 68 fa ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    25d8:	48 89 df             	mov    %rbx,%rdi
    25db:	48 8d 95 70 fd ff ff 	lea    -0x290(%rbp),%rdx
    25e2:	4c 89 f6             	mov    %r14,%rsi
    25e5:	49 89 c5             	mov    %rax,%r13
    25e8:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    25ef:	48 89 85 70 fd ff ff 	mov    %rax,-0x290(%rbp)
    25f6:	48 8b 85 20 fd ff ff 	mov    -0x2e0(%rbp),%rax
    25fd:	48 89 85 80 fd ff ff 	mov    %rax,-0x280(%rbp)
    2604:	e8 07 09 00 00       	call   2f10 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_>
    2609:	e8 32 fa ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    260e:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
    2612:	48 8d 3d 67 3b 00 00 	lea    0x3b67(%rip),%rdi        # 6180 <_ZSt4cout@GLIBCXX_3.4>
    2619:	4c 29 e8             	sub    %r13,%rax
    261c:	c4 e1 ea 2a c0       	vcvtsi2ss %rax,%xmm2,%xmm0
    2621:	c5 fa 5e 05 e7 19 00 	vdivss 0x19e7(%rip),%xmm0,%xmm0        # 4010 <_IO_stdin_used+0x10>
    2628:	00 
    2629:	c5 fa 5a c0          	vcvtss2sd %xmm0,%xmm0,%xmm0
    262d:	e8 ce fb ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2632:	48 89 c7             	mov    %rax,%rdi
    2635:	e8 86 06 00 00       	call   2cc0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    263a:	48 83 bd 38 fd ff ff 	cmpq   $0x48,-0x2c8(%rbp)
    2641:	48 
    2642:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    2648:	48 be 4b 59 86 38 d6 	movabs $0x346dc5d63886594b,%rsi
    264f:	c5 6d 34 
    2652:	77 40                	ja     2694 <main+0x2b4>
    2654:	eb 47                	jmp    269d <main+0x2bd>
    2656:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    265d:	00 00 00 
    2660:	48 81 fb e7 03 00 00 	cmp    $0x3e7,%rbx
    2667:	0f 86 c1 04 00 00    	jbe    2b2e <main+0x74e>
    266d:	48 81 fb 0f 27 00 00 	cmp    $0x270f,%rbx
    2674:	0f 86 bd 04 00 00    	jbe    2b37 <main+0x757>
    267a:	48 89 d8             	mov    %rbx,%rax
    267d:	41 83 c5 04          	add    $0x4,%r13d
    2681:	48 f7 e6             	mul    %rsi
    2684:	48 c1 ea 0b          	shr    $0xb,%rdx
    2688:	48 81 fb 9f 86 01 00 	cmp    $0x1869f,%rbx
    268f:	76 0c                	jbe    269d <main+0x2bd>
    2691:	48 89 d3             	mov    %rdx,%rbx
    2694:	48 83 fb 63          	cmp    $0x63,%rbx
    2698:	77 c6                	ja     2660 <main+0x280>
    269a:	41 ff c5             	inc    %r13d
    269d:	48 8d 9d b0 fd ff ff 	lea    -0x250(%rbp),%rbx
    26a4:	45 89 ee             	mov    %r13d,%r14d
    26a7:	48 8d 85 c0 fd ff ff 	lea    -0x240(%rbp),%rax
    26ae:	48 c7 85 b8 fd ff ff 	movq   $0x0,-0x248(%rbp)
    26b5:	00 00 00 00 
    26b9:	4c 89 f6             	mov    %r14,%rsi
    26bc:	48 89 df             	mov    %rbx,%rdi
    26bf:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
    26c6:	c6 85 c0 fd ff ff 00 	movb   $0x0,-0x240(%rbp)
    26cd:	e8 1e fb ff ff       	call   21f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@plt>
    26d2:	c5 fd 6f 05 46 1a 00 	vmovdqa 0x1a46(%rip),%ymm0        # 4120 <_IO_stdin_used+0x120>
    26d9:	00 
    26da:	41 ff cd             	dec    %r13d
    26dd:	48 81 bd 38 fd ff ff 	cmpq   $0x318,-0x2c8(%rbp)
    26e4:	18 03 00 00 
    26e8:	48 8b bd b0 fd ff ff 	mov    -0x250(%rbp),%rdi
    26ef:	c5 fd 7f 85 d0 fd ff 	vmovdqa %ymm0,-0x230(%rbp)
    26f6:	ff 
    26f7:	c5 fd 6f 05 41 1a 00 	vmovdqa 0x1a41(%rip),%ymm0        # 4140 <_IO_stdin_used+0x140>
    26fe:	00 
    26ff:	c5 fd 7f 85 f0 fd ff 	vmovdqa %ymm0,-0x210(%rbp)
    2706:	ff 
    2707:	c5 fd 6f 05 51 1a 00 	vmovdqa 0x1a51(%rip),%ymm0        # 4160 <_IO_stdin_used+0x160>
    270e:	00 
    270f:	c5 fd 7f 85 10 fe ff 	vmovdqa %ymm0,-0x1f0(%rbp)
    2716:	ff 
    2717:	c5 fd 6f 05 61 1a 00 	vmovdqa 0x1a61(%rip),%ymm0        # 4180 <_IO_stdin_used+0x180>
    271e:	00 
    271f:	c5 fd 7f 85 30 fe ff 	vmovdqa %ymm0,-0x1d0(%rbp)
    2726:	ff 
    2727:	c5 fd 6f 05 71 1a 00 	vmovdqa 0x1a71(%rip),%ymm0        # 41a0 <_IO_stdin_used+0x1a0>
    272e:	00 
    272f:	c5 fd 7f 85 50 fe ff 	vmovdqa %ymm0,-0x1b0(%rbp)
    2736:	ff 
    2737:	c5 fd 6f 05 81 1a 00 	vmovdqa 0x1a81(%rip),%ymm0        # 41c0 <_IO_stdin_used+0x1c0>
    273e:	00 
    273f:	c5 fd 7f 85 70 fe ff 	vmovdqa %ymm0,-0x190(%rbp)
    2746:	ff 
    2747:	c5 f9 6f 05 b1 19 00 	vmovdqa 0x19b1(%rip),%xmm0        # 4100 <_IO_stdin_used+0x100>
    274e:	00 
    274f:	c5 fa 7f 85 89 fe ff 	vmovdqu %xmm0,-0x177(%rbp)
    2756:	ff 
    2757:	76 5f                	jbe    27b8 <main+0x3d8>
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
    2786:	49 89 d7             	mov    %rdx,%r15
    2789:	44 89 ea             	mov    %r13d,%edx
    278c:	48 01 c0             	add    %rax,%rax
    278f:	44 0f b6 84 05 d1 fd 	movzbl -0x22f(%rbp,%rax,1),%r8d
    2796:	ff ff 
    2798:	44 88 04 17          	mov    %r8b,(%rdi,%rdx,1)
    279c:	0f b6 84 05 d0 fd ff 	movzbl -0x230(%rbp,%rax,1),%eax
    27a3:	ff 
    27a4:	41 8d 55 ff          	lea    -0x1(%r13),%edx
    27a8:	41 83 ed 02          	sub    $0x2,%r13d
    27ac:	88 04 17             	mov    %al,(%rdi,%rdx,1)
    27af:	48 81 f9 0f 27 00 00 	cmp    $0x270f,%rcx
    27b6:	77 b0                	ja     2768 <main+0x388>
    27b8:	41 8d 47 30          	lea    0x30(%r15),%eax
    27bc:	49 83 ff 09          	cmp    $0x9,%r15
    27c0:	76 17                	jbe    27d9 <main+0x3f9>
    27c2:	4b 8d 0c 3f          	lea    (%r15,%r15,1),%rcx
    27c6:	0f b6 84 0d d1 fd ff 	movzbl -0x22f(%rbp,%rcx,1),%eax
    27cd:	ff 
    27ce:	88 47 01             	mov    %al,0x1(%rdi)
    27d1:	0f b6 84 0d d0 fd ff 	movzbl -0x230(%rbp,%rcx,1),%eax
    27d8:	ff 
    27d9:	88 07                	mov    %al,(%rdi)
    27db:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    27e2:	31 f6                	xor    %esi,%esi
    27e4:	48 89 df             	mov    %rbx,%rdi
    27e7:	4c 89 b5 b8 fd ff ff 	mov    %r14,-0x248(%rbp)
    27ee:	48 8d 15 57 18 00 00 	lea    0x1857(%rip),%rdx        # 404c <_IO_stdin_used+0x4c>
    27f5:	42 c6 04 30 00       	movb   $0x0,(%rax,%r14,1)
    27fa:	c5 f8 77             	vzeroupper
    27fd:	e8 5e f8 ff ff       	call   2060 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@plt>
    2802:	48 89 c6             	mov    %rax,%rsi
    2805:	4c 89 e7             	mov    %r12,%rdi
    2808:	e8 c3 f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
    280d:	48 8d 35 4d 18 00 00 	lea    0x184d(%rip),%rsi        # 4061 <_IO_stdin_used+0x61>
    2814:	4c 89 e7             	mov    %r12,%rdi
    2817:	e8 64 fa ff ff       	call   2280 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>
    281c:	48 89 c6             	mov    %rax,%rsi
    281f:	48 8d 85 90 fd ff ff 	lea    -0x270(%rbp),%rax
    2826:	4c 8d b5 c8 fe ff ff 	lea    -0x138(%rbp),%r14
    282d:	48 89 c7             	mov    %rax,%rdi
    2830:	48 89 85 20 fd ff ff 	mov    %rax,-0x2e0(%rbp)
    2837:	e8 94 f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
    283c:	4c 89 e7             	mov    %r12,%rdi
    283f:	e8 4c f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2844:	48 89 df             	mov    %rbx,%rdi
    2847:	e8 44 f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    284c:	4c 89 f7             	mov    %r14,%rdi
    284f:	4c 89 b5 10 fd ff ff 	mov    %r14,-0x2f0(%rbp)
    2856:	e8 25 f8 ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
    285b:	48 8d 05 56 33 00 00 	lea    0x3356(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2862:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    2866:	31 f6                	xor    %esi,%esi
    2868:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    286f:	31 c0                	xor    %eax,%eax
    2871:	66 89 45 a8          	mov    %ax,-0x58(%rbp)
    2875:	48 8b 05 d4 33 00 00 	mov    0x33d4(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    287c:	c5 fd 7f 45 b0       	vmovdqa %ymm0,-0x50(%rbp)
    2881:	48 8b 78 e8          	mov    -0x18(%rax),%rdi
    2885:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    288c:	48 8b 05 c5 33 00 00 	mov    0x33c5(%rip),%rax        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2893:	48 c7 45 a0 00 00 00 	movq   $0x0,-0x60(%rbp)
    289a:	00 
    289b:	4c 01 e7             	add    %r12,%rdi
    289e:	48 89 07             	mov    %rax,(%rdi)
    28a1:	c5 f8 77             	vzeroupper
    28a4:	e8 07 f9 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    28a9:	48 8d 05 a0 34 00 00 	lea    0x34a0(%rip),%rax        # 5d50 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    28b0:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    28b7:	48 83 c0 28          	add    $0x28,%rax
    28bb:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    28c2:	48 8d 85 d8 fd ff ff 	lea    -0x228(%rbp),%rax
    28c9:	48 89 c7             	mov    %rax,%rdi
    28cc:	48 89 85 18 fd ff ff 	mov    %rax,-0x2e8(%rbp)
    28d3:	48 89 c3             	mov    %rax,%rbx
    28d6:	e8 85 f8 ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
    28db:	48 89 de             	mov    %rbx,%rsi
    28de:	4c 89 f7             	mov    %r14,%rdi
    28e1:	e8 ca f8 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    28e6:	48 8b b5 90 fd ff ff 	mov    -0x270(%rbp),%rsi
    28ed:	ba 10 00 00 00       	mov    $0x10,%edx
    28f2:	48 89 df             	mov    %rbx,%rdi
    28f5:	e8 36 f8 ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
    28fa:	48 8b 95 d0 fd ff ff 	mov    -0x230(%rbp),%rdx
    2901:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    2905:	4c 01 e7             	add    %r12,%rdi
    2908:	48 85 c0             	test   %rax,%rax
    290b:	0f 84 0d 02 00 00    	je     2b1e <main+0x73e>
    2911:	31 f6                	xor    %esi,%esi
    2913:	e8 08 f9 ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    2918:	48 8b 85 40 fd ff ff 	mov    -0x2c0(%rbp),%rax
    291f:	4c 8d 35 40 17 00 00 	lea    0x1740(%rip),%r14        # 4066 <_IO_stdin_used+0x66>
    2926:	4c 8d 2d 3b 17 00 00 	lea    0x173b(%rip),%r13        # 4068 <_IO_stdin_used+0x68>
    292d:	48 89 c3             	mov    %rax,%rbx
    2930:	48 39 85 48 fd ff ff 	cmp    %rax,-0x2b8(%rbp)
    2937:	74 58                	je     2991 <main+0x5b1>
    2939:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2940:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    2944:	4c 89 e7             	mov    %r12,%rdi
    2947:	c5 f2 5a 03          	vcvtss2sd (%rbx),%xmm1,%xmm0
    294b:	e8 b0 f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2950:	ba 01 00 00 00       	mov    $0x1,%edx
    2955:	4c 89 f6             	mov    %r14,%rsi
    2958:	48 89 c7             	mov    %rax,%rdi
    295b:	49 89 c7             	mov    %rax,%r15
    295e:	e8 0d f8 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2963:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
    2967:	4c 89 ff             	mov    %r15,%rdi
    296a:	c5 f2 5a 43 04       	vcvtss2sd 0x4(%rbx),%xmm1,%xmm0
    296f:	e8 8c f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2974:	48 89 c7             	mov    %rax,%rdi
    2977:	ba 01 00 00 00       	mov    $0x1,%edx
    297c:	4c 89 ee             	mov    %r13,%rsi
    297f:	e8 ec f7 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2984:	48 83 c3 08          	add    $0x8,%rbx
    2988:	48 39 9d 48 fd ff ff 	cmp    %rbx,-0x2b8(%rbp)
    298f:	75 af                	jne    2940 <main+0x560>
    2991:	48 8d 05 e0 33 00 00 	lea    0x33e0(%rip),%rax        # 5d78 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x40>
    2998:	c5 fa 7e 1d 08 34 00 	vmovq  0x3408(%rip),%xmm3        # 5da8 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x70>
    299f:	00 
    29a0:	48 8b bd 18 fd ff ff 	mov    -0x2e8(%rbp),%rdi
    29a7:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    29ae:	48 8d 05 13 33 00 00 	lea    0x3313(%rip),%rax        # 5cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29b5:	c4 e3 e1 22 c0 01    	vpinsrq $0x1,%rax,%xmm3,%xmm0
    29bb:	c5 f9 7f 85 d0 fd ff 	vmovdqa %xmm0,-0x230(%rbp)
    29c2:	ff 
    29c3:	e8 88 f6 ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
    29c8:	48 8d bd 40 fe ff ff 	lea    -0x1c0(%rbp),%rdi
    29cf:	e8 8c f8 ff ff       	call   2260 <_ZNSt12__basic_fileIcED1Ev@plt>
    29d4:	48 8d 05 fd 31 00 00 	lea    0x31fd(%rip),%rax        # 5bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29db:	48 8d bd 10 fe ff ff 	lea    -0x1f0(%rbp),%rdi
    29e2:	48 89 85 d8 fd ff ff 	mov    %rax,-0x228(%rbp)
    29e9:	e8 d2 f7 ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
    29ee:	48 8b 05 5b 32 00 00 	mov    0x325b(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    29f5:	48 8b 0d 5c 32 00 00 	mov    0x325c(%rip),%rcx        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29fc:	48 8b bd 10 fd ff ff 	mov    -0x2f0(%rbp),%rdi
    2a03:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    2a0a:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2a0e:	48 89 8c 05 d0 fd ff 	mov    %rcx,-0x230(%rbp,%rax,1)
    2a15:	ff 
    2a16:	48 8d 05 9b 31 00 00 	lea    0x319b(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2a1d:	48 89 85 c8 fe ff ff 	mov    %rax,-0x138(%rbp)
    2a24:	e8 67 f6 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    2a29:	48 8b bd 20 fd ff ff 	mov    -0x2e0(%rbp),%rdi
    2a30:	e8 5b f7 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2a35:	48 83 bd 40 fd ff ff 	cmpq   $0x0,-0x2c0(%rbp)
    2a3c:	00 
    2a3d:	74 13                	je     2a52 <main+0x672>
    2a3f:	48 8b b5 38 fd ff ff 	mov    -0x2c8(%rbp),%rsi
    2a46:	48 8b bd 40 fd ff ff 	mov    -0x2c0(%rbp),%rdi
    2a4d:	e8 fe f6 ff ff       	call   2150 <_ZdlPvm@plt>
    2a52:	48 8b bd 30 fd ff ff 	mov    -0x2d0(%rbp),%rdi
    2a59:	48 85 ff             	test   %rdi,%rdi
    2a5c:	74 0f                	je     2a6d <main+0x68d>
    2a5e:	48 8b b5 08 fd ff ff 	mov    -0x2f8(%rbp),%rsi
    2a65:	48 29 fe             	sub    %rdi,%rsi
    2a68:	e8 e3 f6 ff ff       	call   2150 <_ZdlPvm@plt>
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
    2a87:	4c 89 e7             	mov    %r12,%rdi
    2a8a:	48 8d b5 b0 fd ff ff 	lea    -0x250(%rbp),%rsi
    2a91:	31 d2                	xor    %edx,%edx
    2a93:	e8 b8 f7 ff ff       	call   2250 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
    2a98:	48 89 85 d0 fd ff ff 	mov    %rax,-0x230(%rbp)
    2a9f:	48 89 c7             	mov    %rax,%rdi
    2aa2:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
    2aa9:	48 89 85 e0 fd ff ff 	mov    %rax,-0x220(%rbp)
    2ab0:	48 89 da             	mov    %rbx,%rdx
    2ab3:	4c 89 ee             	mov    %r13,%rsi
    2ab6:	e8 55 f6 ff ff       	call   2110 <memcpy@plt>
    2abb:	e9 9f f9 ff ff       	jmp    245f <main+0x7f>
    2ac0:	31 d2                	xor    %edx,%edx
    2ac2:	48 89 95 28 fd ff ff 	mov    %rdx,-0x2d8(%rbp)
    2ac9:	48 89 95 20 fd ff ff 	mov    %rdx,-0x2e0(%rbp)
    2ad0:	48 89 95 40 fd ff ff 	mov    %rdx,-0x2c0(%rbp)
    2ad7:	48 89 95 48 fd ff ff 	mov    %rdx,-0x2b8(%rbp)
    2ade:	e9 e2 fa ff ff       	jmp    25c5 <main+0x1e5>
    2ae3:	48 8d 1d b6 37 00 00 	lea    0x37b6(%rip),%rbx        # 62a0 <_ZSt4cerr@GLIBCXX_3.4>
    2aea:	ba 1d 00 00 00       	mov    $0x1d,%edx
    2aef:	48 8d 35 38 15 00 00 	lea    0x1538(%rip),%rsi        # 402e <_IO_stdin_used+0x2e>
    2af6:	48 89 df             	mov    %rbx,%rdi
    2af9:	e8 72 f6 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2afe:	48 89 df             	mov    %rbx,%rdi
    2b01:	e8 ba 01 00 00       	call   2cc0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    2b06:	b8 01 00 00 00       	mov    $0x1,%eax
    2b0b:	e9 5f ff ff ff       	jmp    2a6f <main+0x68f>
    2b10:	48 85 c0             	test   %rax,%rax
    2b13:	0f 84 46 f9 ff ff    	je     245f <main+0x7f>
    2b19:	4c 89 f7             	mov    %r14,%rdi
    2b1c:	eb 92                	jmp    2ab0 <main+0x6d0>
    2b1e:	8b 77 20             	mov    0x20(%rdi),%esi
    2b21:	83 ce 04             	or     $0x4,%esi
    2b24:	e8 f7 f6 ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    2b29:	e9 ea fd ff ff       	jmp    2918 <main+0x538>
    2b2e:	41 83 c5 02          	add    $0x2,%r13d
    2b32:	e9 66 fb ff ff       	jmp    269d <main+0x2bd>
    2b37:	41 83 c5 03          	add    $0x3,%r13d
    2b3b:	e9 5d fb ff ff       	jmp    269d <main+0x2bd>
    2b40:	c5 f8 77             	vzeroupper
    2b43:	e9 76 fa ff ff       	jmp    25be <main+0x1de>
    2b48:	48 8b b5 40 fd ff ff 	mov    -0x2c0(%rbp),%rsi
    2b4f:	31 c0                	xor    %eax,%eax
    2b51:	e9 3a fa ff ff       	jmp    2590 <main+0x1b0>
    2b56:	e9 76 f7 ff ff       	jmp    22d1 <main.cold+0xc>
    2b5b:	49 89 c4             	mov    %rax,%r12
    2b5e:	c5 f8 77             	vzeroupper
    2b61:	e9 ad f7 ff ff       	jmp    2313 <main.cold+0x4e>
    2b66:	48 89 c3             	mov    %rax,%rbx
    2b69:	e9 c4 f7 ff ff       	jmp    2332 <main.cold+0x6d>
    2b6e:	48 89 c7             	mov    %rax,%rdi
    2b71:	e9 d8 f7 ff ff       	jmp    234e <main.cold+0x89>
    2b76:	48 89 c3             	mov    %rax,%rbx
    2b79:	e9 ee f7 ff ff       	jmp    236c <main.cold+0xa7>
    2b7e:	49 89 c4             	mov    %rax,%r12
    2b81:	c5 f8 77             	vzeroupper
    2b84:	e9 05 f8 ff ff       	jmp    238e <main.cold+0xc9>
    2b89:	49 89 c4             	mov    %rax,%r12
    2b8c:	e9 ee f7 ff ff       	jmp    237f <main.cold+0xba>
    2b91:	49 89 c4             	mov    %rax,%r12
    2b94:	e9 35 f8 ff ff       	jmp    23ce <main.cold+0x109>
    2b99:	49 89 c4             	mov    %rax,%r12
    2b9c:	c5 f8 77             	vzeroupper
    2b9f:	e9 67 f7 ff ff       	jmp    230b <main.cold+0x46>
    2ba4:	49 89 c5             	mov    %rax,%r13
    2ba7:	e9 51 f7 ff ff       	jmp    22fd <main.cold+0x38>
    2bac:	49 89 c4             	mov    %rax,%r12
    2baf:	c5 f8 77             	vzeroupper
    2bb2:	e9 f8 f7 ff ff       	jmp    23af <main.cold+0xea>
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
    2bdb:	ff 15 e7 33 00 00    	call   *0x33e7(%rip)        # 5fc8 <__libc_start_main@GLIBC_2.34>
    2be1:	f4                   	hlt
    2be2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2be9:	00 00 00 
    2bec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002bf0 <deregister_tm_clones>:
    2bf0:	48 8d 3d 59 35 00 00 	lea    0x3559(%rip),%rdi        # 6150 <__TMC_END__>
    2bf7:	48 8d 05 52 35 00 00 	lea    0x3552(%rip),%rax        # 6150 <__TMC_END__>
    2bfe:	48 39 f8             	cmp    %rdi,%rax
    2c01:	74 15                	je     2c18 <deregister_tm_clones+0x28>
    2c03:	48 8b 05 c6 33 00 00 	mov    0x33c6(%rip),%rax        # 5fd0 <_ITM_deregisterTMCloneTable@Base>
    2c0a:	48 85 c0             	test   %rax,%rax
    2c0d:	74 09                	je     2c18 <deregister_tm_clones+0x28>
    2c0f:	ff e0                	jmp    *%rax
    2c11:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2c18:	c3                   	ret
    2c19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002c20 <register_tm_clones>:
    2c20:	48 8d 3d 29 35 00 00 	lea    0x3529(%rip),%rdi        # 6150 <__TMC_END__>
    2c27:	48 8d 35 22 35 00 00 	lea    0x3522(%rip),%rsi        # 6150 <__TMC_END__>
    2c2e:	48 29 fe             	sub    %rdi,%rsi
    2c31:	48 89 f0             	mov    %rsi,%rax
    2c34:	48 c1 ee 3f          	shr    $0x3f,%rsi
    2c38:	48 c1 f8 03          	sar    $0x3,%rax
    2c3c:	48 01 c6             	add    %rax,%rsi
    2c3f:	48 d1 fe             	sar    $1,%rsi
    2c42:	74 14                	je     2c58 <register_tm_clones+0x38>
    2c44:	48 8b 05 95 33 00 00 	mov    0x3395(%rip),%rax        # 5fe0 <_ITM_registerTMCloneTable@Base>
    2c4b:	48 85 c0             	test   %rax,%rax
    2c4e:	74 08                	je     2c58 <register_tm_clones+0x38>
    2c50:	ff e0                	jmp    *%rax
    2c52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2c58:	c3                   	ret
    2c59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002c60 <__do_global_dtors_aux>:
    2c60:	f3 0f 1e fa          	endbr64
    2c64:	80 3d 45 37 00 00 00 	cmpb   $0x0,0x3745(%rip)        # 63b0 <completed.0>
    2c6b:	75 2b                	jne    2c98 <__do_global_dtors_aux+0x38>
    2c6d:	55                   	push   %rbp
    2c6e:	48 83 3d 4a 33 00 00 	cmpq   $0x0,0x334a(%rip)        # 5fc0 <__cxa_finalize@GLIBC_2.2.5>
    2c75:	00 
    2c76:	48 89 e5             	mov    %rsp,%rbp
    2c79:	74 0c                	je     2c87 <__do_global_dtors_aux+0x27>
    2c7b:	48 8b 3d be 34 00 00 	mov    0x34be(%rip),%rdi        # 6140 <__dso_handle>
    2c82:	e8 19 f6 ff ff       	call   22a0 <__cxa_finalize@plt>
    2c87:	e8 64 ff ff ff       	call   2bf0 <deregister_tm_clones>
    2c8c:	c6 05 1d 37 00 00 01 	movb   $0x1,0x371d(%rip)        # 63b0 <completed.0>
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
    2cc0:	55                   	push   %rbp
    2cc1:	53                   	push   %rbx
    2cc2:	48 83 ec 08          	sub    $0x8,%rsp
    2cc6:	48 8b 07             	mov    (%rdi),%rax
    2cc9:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2ccd:	48 8b ac 07 f0 00 00 	mov    0xf0(%rdi,%rax,1),%rbp
    2cd4:	00 
    2cd5:	48 85 ed             	test   %rbp,%rbp
    2cd8:	0f 84 e2 f5 ff ff    	je     22c0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0.cold>
    2cde:	80 7d 38 00          	cmpb   $0x0,0x38(%rbp)
    2ce2:	48 89 fb             	mov    %rdi,%rbx
    2ce5:	74 1a                	je     2d01 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x41>
    2ce7:	0f be 75 43          	movsbl 0x43(%rbp),%esi
    2ceb:	48 89 df             	mov    %rbx,%rdi
    2cee:	e8 3d f3 ff ff       	call   2030 <_ZNSo3putEc@plt>
    2cf3:	48 83 c4 08          	add    $0x8,%rsp
    2cf7:	5b                   	pop    %rbx
    2cf8:	48 89 c7             	mov    %rax,%rdi
    2cfb:	5d                   	pop    %rbp
    2cfc:	e9 df f3 ff ff       	jmp    20e0 <_ZNSo5flushEv@plt>
    2d01:	48 89 ef             	mov    %rbp,%rdi
    2d04:	e8 77 f4 ff ff       	call   2180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
    2d09:	48 8b 45 00          	mov    0x0(%rbp),%rax
    2d0d:	be 0a 00 00 00       	mov    $0xa,%esi
    2d12:	48 89 ef             	mov    %rbp,%rdi
    2d15:	ff 50 30             	call   *0x30(%rax)
    2d18:	0f be f0             	movsbl %al,%esi
    2d1b:	eb ce                	jmp    2ceb <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x2b>
    2d1d:	0f 1f 00             	nopl   (%rax)

0000000000002d20 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_>:
    2d20:	41 56                	push   %r14
    2d22:	41 55                	push   %r13
    2d24:	41 54                	push   %r12
    2d26:	55                   	push   %rbp
    2d27:	53                   	push   %rbx
    2d28:	48 83 ec 20          	sub    $0x20,%rsp
    2d2c:	48 85 ff             	test   %rdi,%rdi
    2d2f:	0f 84 03 01 00 00    	je     2e38 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x118>
    2d35:	48 8b 1e             	mov    (%rsi),%rbx
    2d38:	4c 8b 2a             	mov    (%rdx),%r13
    2d3b:	48 89 fd             	mov    %rdi,%rbp
    2d3e:	0f 88 01 01 00 00    	js     2e45 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x125>
    2d44:	c5 c1 57 ff          	vxorpd %xmm7,%xmm7,%xmm7
    2d48:	c4 e1 c3 2a c7       	vcvtsi2sd %rdi,%xmm7,%xmm0
    2d4d:	c5 f9 13 44 24 18    	vmovlpd %xmm0,0x18(%rsp)
    2d53:	45 31 e4             	xor    %r12d,%r12d
    2d56:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2d5d:	00 00 00 
    2d60:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    2d64:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
    2d68:	45 31 f6             	xor    %r14d,%r14d
    2d6b:	c4 c1 4b 2a c4       	vcvtsi2sd %r12d,%xmm6,%xmm0
    2d70:	c5 fb 59 15 68 13 00 	vmulsd 0x1368(%rip),%xmm0,%xmm2        # 40e0 <_IO_stdin_used+0xe0>
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
    2dad:	c5 fa 11 4c 24 04    	vmovss %xmm1,0x4(%rsp)
    2db3:	c5 f0 57 05 35 13 00 	vxorps 0x1335(%rip),%xmm1,%xmm0        # 40f0 <_IO_stdin_used+0xf0>
    2dba:	00 
    2dbb:	e8 60 f3 ff ff       	call   2120 <sinf@plt>
    2dc0:	c5 fa 10 4c 24 04    	vmovss 0x4(%rsp),%xmm1
    2dc6:	c5 fa 11 04 24       	vmovss %xmm0,(%rsp)
    2dcb:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    2dcf:	e8 2c f3 ff ff       	call   2100 <cosf@plt>
    2dd4:	c5 fa 10 1c 24       	vmovss (%rsp),%xmm3
    2dd9:	c4 a1 7a 10 4c f3 04 	vmovss 0x4(%rbx,%r14,8),%xmm1
    2de0:	c4 a1 7a 10 3c f3    	vmovss (%rbx,%r14,8),%xmm7
    2de6:	c5 fa 10 64 24 08    	vmovss 0x8(%rsp),%xmm4
    2dec:	c5 f2 59 d0          	vmulss %xmm0,%xmm1,%xmm2
    2df0:	c5 fa 10 6c 24 0c    	vmovss 0xc(%rsp),%xmm5
    2df6:	c5 f2 59 f3          	vmulss %xmm3,%xmm1,%xmm6
    2dfa:	c4 e2 41 b9 d3       	vfmadd231ss %xmm3,%xmm7,%xmm2
    2dff:	c4 e2 41 bb f0       	vfmsub231ss %xmm0,%xmm7,%xmm6
    2e04:	c5 f8 2e d6          	vucomiss %xmm6,%xmm2
    2e08:	7a 62                	jp     2e6c <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x14c>
    2e0a:	49 ff c6             	inc    %r14
    2e0d:	c5 d2 58 ee          	vaddss %xmm6,%xmm5,%xmm5
    2e11:	c5 da 58 e2          	vaddss %xmm2,%xmm4,%xmm4
    2e15:	4c 39 f5             	cmp    %r14,%rbp
    2e18:	0f 85 6a ff ff ff    	jne    2d88 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x68>
    2e1e:	c4 81 7a 11 6c e5 00 	vmovss %xmm5,0x0(%r13,%r12,8)
    2e25:	c4 81 7a 11 64 e5 04 	vmovss %xmm4,0x4(%r13,%r12,8)
    2e2c:	49 ff c4             	inc    %r12
    2e2f:	4c 39 e5             	cmp    %r12,%rbp
    2e32:	0f 85 28 ff ff ff    	jne    2d60 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
    2e38:	48 83 c4 20          	add    $0x20,%rsp
    2e3c:	5b                   	pop    %rbx
    2e3d:	5d                   	pop    %rbp
    2e3e:	41 5c                	pop    %r12
    2e40:	41 5d                	pop    %r13
    2e42:	41 5e                	pop    %r14
    2e44:	c3                   	ret
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
    2e82:	e8 e9 f3 ff ff       	call   2270 <__mulsc3@plt>
    2e87:	c5 fa 10 6c 24 04    	vmovss 0x4(%rsp),%xmm5
    2e8d:	c5 fa 10 24 24       	vmovss (%rsp),%xmm4
    2e92:	c4 e1 f9 7e c0       	vmovq  %xmm0,%rax
    2e97:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    2e9b:	48 c1 e8 20          	shr    $0x20,%rax
    2e9f:	c5 d2 58 e9          	vaddss %xmm1,%xmm5,%xmm5
    2ea3:	c5 f9 6e c0          	vmovd  %eax,%xmm0
    2ea7:	c5 da 58 e0          	vaddss %xmm0,%xmm4,%xmm4
    2eab:	4c 39 f5             	cmp    %r14,%rbp
    2eae:	0f 85 d4 fe ff ff    	jne    2d88 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x68>
    2eb4:	e9 65 ff ff ff       	jmp    2e1e <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0xfe>
    2eb9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002ec0 <_Z11bit_reversemi>:
    2ec0:	85 f6                	test   %esi,%esi
    2ec2:	7e 3c                	jle    2f00 <_Z11bit_reversemi+0x40>
    2ec4:	31 d2                	xor    %edx,%edx
    2ec6:	31 c0                	xor    %eax,%eax
    2ec8:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2ecf:	00 00 00 00 
    2ed3:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2eda:	00 00 00 00 
    2ede:	66 90                	xchg   %ax,%ax
    2ee0:	48 89 f9             	mov    %rdi,%rcx
    2ee3:	48 01 c0             	add    %rax,%rax
    2ee6:	ff c2                	inc    %edx
    2ee8:	48 d1 ef             	shr    $1,%rdi
    2eeb:	83 e1 01             	and    $0x1,%ecx
    2eee:	48 09 c8             	or     %rcx,%rax
    2ef1:	39 d6                	cmp    %edx,%esi
    2ef3:	75 eb                	jne    2ee0 <_Z11bit_reversemi+0x20>
    2ef5:	c3                   	ret
    2ef6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2efd:	00 00 00 
    2f00:	31 c0                	xor    %eax,%eax
    2f02:	c3                   	ret
    2f03:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2f0a:	00 00 00 00 
    2f0e:	66 90                	xchg   %ax,%ax

0000000000002f10 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_>:
    2f10:	41 57                	push   %r15
    2f12:	41 56                	push   %r14
    2f14:	41 55                	push   %r13
    2f16:	41 54                	push   %r12
    2f18:	55                   	push   %rbp
    2f19:	53                   	push   %rbx
    2f1a:	48 83 ec 58          	sub    $0x58,%rsp
    2f1e:	48 85 ff             	test   %rdi,%rdi
    2f21:	0f 84 2a 02 00 00    	je     3151 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x241>
    2f27:	48 89 fd             	mov    %rdi,%rbp
    2f2a:	49 89 f4             	mov    %rsi,%r12
    2f2d:	49 89 d5             	mov    %rdx,%r13
    2f30:	0f 88 31 02 00 00    	js     3167 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x257>
    2f36:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    2f3a:	c4 e1 cb 2a c7       	vcvtsi2sd %rdi,%xmm6,%xmm0
    2f3f:	c4 e1 f9 7e c3       	vmovq  %xmm0,%rbx
    2f44:	45 31 ff             	xor    %r15d,%r15d
    2f47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2f4e:	00 00 
    2f50:	c4 e1 f9 6e c3       	vmovq  %rbx,%xmm0
    2f55:	e8 e6 f2 ff ff       	call   2240 <log2@plt>
    2f5a:	c5 fb 2c f8          	vcvttsd2si %xmm0,%edi
    2f5e:	85 ff                	test   %edi,%edi
    2f60:	0f 8e fa 01 00 00    	jle    3160 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x250>
    2f66:	4c 89 f9             	mov    %r15,%rcx
    2f69:	31 d2                	xor    %edx,%edx
    2f6b:	31 c0                	xor    %eax,%eax
    2f6d:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2f74:	00 00 00 00 
    2f78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2f7f:	00 
    2f80:	48 89 ce             	mov    %rcx,%rsi
    2f83:	48 01 c0             	add    %rax,%rax
    2f86:	ff c2                	inc    %edx
    2f88:	48 d1 e9             	shr    $1,%rcx
    2f8b:	83 e6 01             	and    $0x1,%esi
    2f8e:	48 09 f0             	or     %rsi,%rax
    2f91:	39 d7                	cmp    %edx,%edi
    2f93:	75 eb                	jne    2f80 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x70>
    2f95:	48 c1 e0 03          	shl    $0x3,%rax
    2f99:	49 8b 14 24          	mov    (%r12),%rdx
    2f9d:	4d 8b 45 00          	mov    0x0(%r13),%r8
    2fa1:	c5 fa 10 04 02       	vmovss (%rdx,%rax,1),%xmm0
    2fa6:	c4 81 7a 11 04 f8    	vmovss %xmm0,(%r8,%r15,8)
    2fac:	c5 fa 10 44 02 04    	vmovss 0x4(%rdx,%rax,1),%xmm0
    2fb2:	c4 81 7a 11 44 f8 04 	vmovss %xmm0,0x4(%r8,%r15,8)
    2fb9:	49 ff c7             	inc    %r15
    2fbc:	4c 39 fd             	cmp    %r15,%rbp
    2fbf:	75 8f                	jne    2f50 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
    2fc1:	49 83 ff 01          	cmp    $0x1,%r15
    2fc5:	0f 84 86 01 00 00    	je     3151 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x241>
    2fcb:	c5 fa 10 2d 31 10 00 	vmovss 0x1031(%rip),%xmm5        # 4004 <_IO_stdin_used+0x4>
    2fd2:	00 
    2fd3:	c5 fa 10 25 2d 10 00 	vmovss 0x102d(%rip),%xmm4        # 4008 <_IO_stdin_used+0x8>
    2fda:	00 
    2fdb:	bf 02 00 00 00       	mov    $0x2,%edi
    2fe0:	c5 fa 10 35 24 10 00 	vmovss 0x1024(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    2fe7:	00 
    2fe8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2fef:	00 
    2ff0:	41 89 fd             	mov    %edi,%r13d
    2ff3:	41 d1 fd             	sar    $1,%r13d
    2ff6:	83 ff 01             	cmp    $0x1,%edi
    2ff9:	0f 8e e1 00 00 00    	jle    30e0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1d0>
    2fff:	48 63 d7             	movslq %edi,%rdx
    3002:	49 63 cd             	movslq %r13d,%rcx
    3005:	4c 89 c6             	mov    %r8,%rsi
    3008:	48 8d 04 d5 00 00 00 	lea    0x0(,%rdx,8),%rax
    300f:	00 
    3010:	49 8d 0c c8          	lea    (%r8,%rcx,8),%rcx
    3014:	49 89 d6             	mov    %rdx,%r14
    3017:	48 29 c6             	sub    %rax,%rsi
    301a:	48 29 c1             	sub    %rax,%rcx
    301d:	0f 1f 00             	nopl   (%rax)
    3020:	4a 8d 1c f5 00 00 00 	lea    0x0(,%r14,8),%rbx
    3027:	00 
    3028:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
    302c:	c5 f8 28 d6          	vmovaps %xmm6,%xmm2
    3030:	45 31 e4             	xor    %r12d,%r12d
    3033:	48 8d 2c 33          	lea    (%rbx,%rsi,1),%rbp
    3037:	48 01 cb             	add    %rcx,%rbx
    303a:	eb 0c                	jmp    3048 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x138>
    303c:	0f 1f 40 00          	nopl   0x0(%rax)
    3040:	c5 f8 28 d8          	vmovaps %xmm0,%xmm3
    3044:	c5 f8 28 d1          	vmovaps %xmm1,%xmm2
    3048:	c5 fa 10 4b 04       	vmovss 0x4(%rbx),%xmm1
    304d:	c5 7a 10 13          	vmovss (%rbx),%xmm10
    3051:	c5 7a 10 45 00       	vmovss 0x0(%rbp),%xmm8
    3056:	c5 fa 10 7d 04       	vmovss 0x4(%rbp),%xmm7
    305b:	c5 f2 59 c2          	vmulss %xmm2,%xmm1,%xmm0
    305f:	c5 72 59 cb          	vmulss %xmm3,%xmm1,%xmm9
    3063:	c4 e2 29 b9 c3       	vfmadd231ss %xmm3,%xmm10,%xmm0
    3068:	c4 62 29 bb ca       	vfmsub231ss %xmm2,%xmm10,%xmm9
    306d:	c4 c1 78 2e c1       	vucomiss %xmm9,%xmm0
    3072:	0f 8a 86 01 00 00    	jp     31fe <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x2ee>
    3078:	c4 c1 32 58 c8       	vaddss %xmm8,%xmm9,%xmm1
    307d:	c4 41 3a 5c c1       	vsubss %xmm9,%xmm8,%xmm8
    3082:	c5 fa 11 4d 00       	vmovss %xmm1,0x0(%rbp)
    3087:	c5 fa 58 cf          	vaddss %xmm7,%xmm0,%xmm1
    308b:	c5 c2 5c f8          	vsubss %xmm0,%xmm7,%xmm7
    308f:	c5 d2 59 c2          	vmulss %xmm2,%xmm5,%xmm0
    3093:	c5 fa 11 4d 04       	vmovss %xmm1,0x4(%rbp)
    3098:	c5 d2 59 cb          	vmulss %xmm3,%xmm5,%xmm1
    309c:	c5 7a 11 03          	vmovss %xmm8,(%rbx)
    30a0:	c4 e2 59 b9 c3       	vfmadd231ss %xmm3,%xmm4,%xmm0
    30a5:	c5 fa 11 7b 04       	vmovss %xmm7,0x4(%rbx)
    30aa:	c4 e2 59 bb ca       	vfmsub231ss %xmm2,%xmm4,%xmm1
    30af:	c5 f8 2e c1          	vucomiss %xmm1,%xmm0
    30b3:	0f 8a d4 00 00 00    	jp     318d <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x27d>
    30b9:	41 ff c4             	inc    %r12d
    30bc:	48 83 c5 08          	add    $0x8,%rbp
    30c0:	48 83 c3 08          	add    $0x8,%rbx
    30c4:	45 39 e5             	cmp    %r12d,%r13d
    30c7:	0f 8f 73 ff ff ff    	jg     3040 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x130>
    30cd:	4a 8d 04 32          	lea    (%rdx,%r14,1),%rax
    30d1:	4d 39 fe             	cmp    %r15,%r14
    30d4:	73 0a                	jae    30e0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1d0>
    30d6:	49 89 c6             	mov    %rax,%r14
    30d9:	e9 42 ff ff ff       	jmp    3020 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x110>
    30de:	66 90                	xchg   %ax,%ax
    30e0:	01 ff                	add    %edi,%edi
    30e2:	4c 89 04 24          	mov    %r8,(%rsp)
    30e6:	48 63 c7             	movslq %edi,%rax
    30e9:	49 39 c7             	cmp    %rax,%r15
    30ec:	72 63                	jb     3151 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x241>
    30ee:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    30f2:	89 7c 24 10          	mov    %edi,0x10(%rsp)
    30f6:	c5 cb 2a c7          	vcvtsi2sd %edi,%xmm6,%xmm0
    30fa:	c5 fb 10 35 de 0f 00 	vmovsd 0xfde(%rip),%xmm6        # 40e0 <_IO_stdin_used+0xe0>
    3101:	00 
    3102:	c5 cb 5e c0          	vdivsd %xmm0,%xmm6,%xmm0
    3106:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
    310a:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
    310e:	c5 fa 11 4c 24 30    	vmovss %xmm1,0x30(%rsp)
    3114:	e8 e7 ef ff ff       	call   2100 <cosf@plt>
    3119:	c5 fa 10 4c 24 30    	vmovss 0x30(%rsp),%xmm1
    311f:	c5 fa 11 44 24 0c    	vmovss %xmm0,0xc(%rsp)
    3125:	c5 f0 57 05 c3 0f 00 	vxorps 0xfc3(%rip),%xmm1,%xmm0        # 40f0 <_IO_stdin_used+0xf0>
    312c:	00 
    312d:	e8 ee ef ff ff       	call   2120 <sinf@plt>
    3132:	c5 fa 10 64 24 0c    	vmovss 0xc(%rsp),%xmm4
    3138:	8b 7c 24 10          	mov    0x10(%rsp),%edi
    313c:	4c 8b 04 24          	mov    (%rsp),%r8
    3140:	c5 fa 10 35 c4 0e 00 	vmovss 0xec4(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    3147:	00 
    3148:	c5 f8 28 e8          	vmovaps %xmm0,%xmm5
    314c:	e9 9f fe ff ff       	jmp    2ff0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0xe0>
    3151:	48 83 c4 58          	add    $0x58,%rsp
    3155:	5b                   	pop    %rbx
    3156:	5d                   	pop    %rbp
    3157:	41 5c                	pop    %r12
    3159:	41 5d                	pop    %r13
    315b:	41 5e                	pop    %r14
    315d:	41 5f                	pop    %r15
    315f:	c3                   	ret
    3160:	31 c0                	xor    %eax,%eax
    3162:	e9 32 fe ff ff       	jmp    2f99 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x89>
    3167:	48 89 f8             	mov    %rdi,%rax
    316a:	48 89 fa             	mov    %rdi,%rdx
    316d:	c5 c9 57 f6          	vxorpd %xmm6,%xmm6,%xmm6
    3171:	48 d1 e8             	shr    $1,%rax
    3174:	83 e2 01             	and    $0x1,%edx
    3177:	48 09 d0             	or     %rdx,%rax
    317a:	c4 e1 cb 2a c0       	vcvtsi2sd %rax,%xmm6,%xmm0
    317f:	c5 fb 58 f0          	vaddsd %xmm0,%xmm0,%xmm6
    3183:	c4 e1 f9 7e f3       	vmovq  %xmm6,%rbx
    3188:	e9 b7 fd ff ff       	jmp    2f44 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x34>
    318d:	c5 f8 28 c4          	vmovaps %xmm4,%xmm0
    3191:	c5 f8 28 cd          	vmovaps %xmm5,%xmm1
    3195:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
    319a:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
    319f:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
    31a4:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
    31a9:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    31ad:	c5 fa 11 6c 24 0c    	vmovss %xmm5,0xc(%rsp)
    31b3:	c5 fa 11 24 24       	vmovss %xmm4,(%rsp)
    31b8:	e8 b3 f0 ff ff       	call   2270 <__mulsc3@plt>
    31bd:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    31c2:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    31c7:	c5 f9 6f f0          	vmovdqa %xmm0,%xmm6
    31cb:	c5 f9 6f c8          	vmovdqa %xmm0,%xmm1
    31cf:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    31d4:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    31d9:	c5 c8 c6 f6 55       	vshufps $0x55,%xmm6,%xmm6,%xmm6
    31de:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    31e2:	c5 f9 6f c6          	vmovdqa %xmm6,%xmm0
    31e6:	c5 fa 10 6c 24 0c    	vmovss 0xc(%rsp),%xmm5
    31ec:	c5 fa 10 35 18 0e 00 	vmovss 0xe18(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    31f3:	00 
    31f4:	c5 fa 10 24 24       	vmovss (%rsp),%xmm4
    31f9:	e9 bb fe ff ff       	jmp    30b9 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1a9>
    31fe:	c5 78 29 d0          	vmovaps %xmm10,%xmm0
    3202:	48 89 54 24 48       	mov    %rdx,0x48(%rsp)
    3207:	48 89 74 24 40       	mov    %rsi,0x40(%rsp)
    320c:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
    3211:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
    3216:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    321a:	c5 fa 11 6c 24 34    	vmovss %xmm5,0x34(%rsp)
    3220:	c5 fa 11 64 24 28    	vmovss %xmm4,0x28(%rsp)
    3226:	c5 7a 11 44 24 20    	vmovss %xmm8,0x20(%rsp)
    322c:	c5 fa 11 7c 24 18    	vmovss %xmm7,0x18(%rsp)
    3232:	c5 fa 11 5c 24 0c    	vmovss %xmm3,0xc(%rsp)
    3238:	c5 fa 11 14 24       	vmovss %xmm2,(%rsp)
    323d:	e8 2e f0 ff ff       	call   2270 <__mulsc3@plt>
    3242:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
    3247:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    324c:	c5 f9 6f f0          	vmovdqa %xmm0,%xmm6
    3250:	c5 79 6f c8          	vmovdqa %xmm0,%xmm9
    3254:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    3258:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    325d:	c5 c8 c6 f6 55       	vshufps $0x55,%xmm6,%xmm6,%xmm6
    3262:	c5 fa 10 6c 24 34    	vmovss 0x34(%rsp),%xmm5
    3268:	c5 f9 6f c6          	vmovdqa %xmm6,%xmm0
    326c:	c5 fa 10 64 24 28    	vmovss 0x28(%rsp),%xmm4
    3272:	c5 fa 10 35 92 0d 00 	vmovss 0xd92(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    3279:	00 
    327a:	c5 7a 10 44 24 20    	vmovss 0x20(%rsp),%xmm8
    3280:	c5 fa 10 7c 24 18    	vmovss 0x18(%rsp),%xmm7
    3286:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    328b:	c5 fa 10 5c 24 0c    	vmovss 0xc(%rsp),%xmm3
    3291:	c5 fa 10 14 24       	vmovss (%rsp),%xmm2
    3296:	e9 dd fd ff ff       	jmp    3078 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x168>
    329b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000032a0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>:
    32a0:	55                   	push   %rbp
    32a1:	48 89 e5             	mov    %rsp,%rbp
    32a4:	41 57                	push   %r15
    32a6:	41 56                	push   %r14
    32a8:	41 55                	push   %r13
    32aa:	41 54                	push   %r12
    32ac:	49 89 f4             	mov    %rsi,%r12
    32af:	53                   	push   %rbx
    32b0:	48 89 fb             	mov    %rdi,%rbx
    32b3:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    32b7:	48 81 ec 60 02 00 00 	sub    $0x260,%rsp
    32be:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
    32c5:	00 
    32c6:	4c 8d 6c 24 40       	lea    0x40(%rsp),%r13
    32cb:	48 89 c7             	mov    %rax,%rdi
    32ce:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    32d3:	e8 a8 ed ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
    32d8:	4c 8b 3d b1 2a 00 00 	mov    0x2ab1(%rip),%r15        # 5d90 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    32df:	31 d2                	xor    %edx,%edx
    32e1:	31 f6                	xor    %esi,%esi
    32e3:	48 8d 0d ce 28 00 00 	lea    0x28ce(%rip),%rcx        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    32ea:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    32ee:	66 89 94 24 20 02 00 	mov    %dx,0x220(%rsp)
    32f5:	00 
    32f6:	c5 fe 7f 84 24 28 02 	vmovdqu %ymm0,0x228(%rsp)
    32fd:	00 00 
    32ff:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    3303:	48 89 8c 24 40 01 00 	mov    %rcx,0x140(%rsp)
    330a:	00 
    330b:	48 8b 0d 86 2a 00 00 	mov    0x2a86(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3312:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
    3319:	00 00 00 00 00 
    331e:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    3323:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    3328:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    332f:	00 00 
    3331:	49 8b 4f e8          	mov    -0x18(%r15),%rcx
    3335:	4c 01 e9             	add    %r13,%rcx
    3338:	48 89 cf             	mov    %rcx,%rdi
    333b:	c5 f8 77             	vzeroupper
    333e:	e8 6d ee ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    3343:	48 8d 0d 36 29 00 00 	lea    0x2936(%rip),%rcx        # 5c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    334a:	4c 8d 74 24 50       	lea    0x50(%rsp),%r14
    334f:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
    3354:	4c 89 f7             	mov    %r14,%rdi
    3357:	48 83 c1 28          	add    $0x28,%rcx
    335b:	48 89 8c 24 40 01 00 	mov    %rcx,0x140(%rsp)
    3362:	00 
    3363:	4c 89 34 24          	mov    %r14,(%rsp)
    3367:	e8 f4 ed ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
    336c:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    3371:	4c 89 f6             	mov    %r14,%rsi
    3374:	e8 37 ee ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    3379:	49 8b 34 24          	mov    (%r12),%rsi
    337d:	ba 08 00 00 00       	mov    $0x8,%edx
    3382:	4c 89 f7             	mov    %r14,%rdi
    3385:	e8 a6 ed ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
    338a:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    338f:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    3393:	4c 01 ef             	add    %r13,%rdi
    3396:	48 85 c0             	test   %rax,%rax
    3399:	0f 84 31 02 00 00    	je     35d0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x330>
    339f:	31 f6                	xor    %esi,%esi
    33a1:	e8 7a ee ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    33a6:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
    33ad:	00 
    33ae:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    33b2:	4c 8d 74 24 38       	lea    0x38(%rsp),%r14
    33b7:	c5 fa 7f 03          	vmovdqu %xmm0,(%rbx)
    33bb:	4c 89 f6             	mov    %r14,%rsi
    33be:	4c 89 ef             	mov    %r13,%rdi
    33c1:	e8 ca ee ff ff       	call   2290 <_ZNSi10_M_extractIfEERSiRT_@plt>
    33c6:	48 89 c7             	mov    %rax,%rdi
    33c9:	48 8d 74 24 3c       	lea    0x3c(%rsp),%rsi
    33ce:	e8 bd ee ff ff       	call   2290 <_ZNSi10_M_extractIfEERSiRT_@plt>
    33d3:	48 8b 10             	mov    (%rax),%rdx
    33d6:	48 8b 52 e8          	mov    -0x18(%rdx),%rdx
    33da:	f6 44 10 20 05       	testb  $0x5,0x20(%rax,%rdx,1)
    33df:	0f 85 4b 01 00 00    	jne    3530 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x290>
    33e5:	48 8b 53 08          	mov    0x8(%rbx),%rdx
    33e9:	48 3b 53 10          	cmp    0x10(%rbx),%rdx
    33ed:	74 31                	je     3420 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x180>
    33ef:	c5 fa 10 44 24 38    	vmovss 0x38(%rsp),%xmm0
    33f5:	48 83 c2 08          	add    $0x8,%rdx
    33f9:	4c 89 f6             	mov    %r14,%rsi
    33fc:	4c 89 ef             	mov    %r13,%rdi
    33ff:	c4 e3 79 21 44 24 3c 	vinsertps $0x10,0x3c(%rsp),%xmm0,%xmm0
    3406:	10 
    3407:	c5 f8 13 42 f8       	vmovlps %xmm0,-0x8(%rdx)
    340c:	48 89 53 08          	mov    %rdx,0x8(%rbx)
    3410:	e8 7b ee ff ff       	call   2290 <_ZNSi10_M_extractIfEERSiRT_@plt>
    3415:	eb af                	jmp    33c6 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x126>
    3417:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    341e:	00 00 
    3420:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    3427:	ff ff 0f 
    342a:	48 8b 0b             	mov    (%rbx),%rcx
    342d:	48 89 d6             	mov    %rdx,%rsi
    3430:	48 29 ce             	sub    %rcx,%rsi
    3433:	49 89 f4             	mov    %rsi,%r12
    3436:	49 c1 fc 03          	sar    $0x3,%r12
    343a:	49 39 c4             	cmp    %rax,%r12
    343d:	0f 84 1c 02 00 00    	je     365f <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3bf>
    3443:	4d 85 e4             	test   %r12,%r12
    3446:	b8 01 00 00 00       	mov    $0x1,%eax
    344b:	49 0f 45 c4          	cmovne %r12,%rax
    344f:	49 01 c4             	add    %rax,%r12
    3452:	0f 82 88 01 00 00    	jb     35e0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x340>
    3458:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    345f:	ff ff 0f 
    3462:	49 39 c4             	cmp    %rax,%r12
    3465:	4c 0f 47 e0          	cmova  %rax,%r12
    3469:	49 c1 e4 03          	shl    $0x3,%r12
    346d:	4c 89 e7             	mov    %r12,%rdi
    3470:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
    3475:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    347a:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
    347f:	e8 bc ec ff ff       	call   2140 <_Znwm@plt>
    3484:	c5 fa 10 44 24 38    	vmovss 0x38(%rsp),%xmm0
    348a:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    348f:	49 89 c0             	mov    %rax,%r8
    3492:	c4 e3 79 21 44 24 3c 	vinsertps $0x10,0x3c(%rsp),%xmm0,%xmm0
    3499:	10 
    349a:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    349f:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    34a4:	c5 f8 13 04 30       	vmovlps %xmm0,(%rax,%rsi,1)
    34a9:	48 39 ca             	cmp    %rcx,%rdx
    34ac:	74 32                	je     34e0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x240>
    34ae:	48 29 ca             	sub    %rcx,%rdx
    34b1:	48 89 ce             	mov    %rcx,%rsi
    34b4:	48 01 c2             	add    %rax,%rdx
    34b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    34be:	00 00 
    34c0:	c5 fa 10 06          	vmovss (%rsi),%xmm0
    34c4:	48 83 c0 08          	add    $0x8,%rax
    34c8:	48 83 c6 08          	add    $0x8,%rsi
    34cc:	c5 fa 11 40 f8       	vmovss %xmm0,-0x8(%rax)
    34d1:	c5 fa 10 46 fc       	vmovss -0x4(%rsi),%xmm0
    34d6:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
    34db:	48 39 d0             	cmp    %rdx,%rax
    34de:	75 e0                	jne    34c0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x220>
    34e0:	48 83 c0 08          	add    $0x8,%rax
    34e4:	c4 c1 f9 6e c8       	vmovq  %r8,%xmm1
    34e9:	c4 e3 f1 22 c0 01    	vpinsrq $0x1,%rax,%xmm1,%xmm0
    34ef:	48 85 c9             	test   %rcx,%rcx
    34f2:	74 25                	je     3519 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x279>
    34f4:	48 8b 73 10          	mov    0x10(%rbx),%rsi
    34f8:	48 89 cf             	mov    %rcx,%rdi
    34fb:	4c 89 44 24 20       	mov    %r8,0x20(%rsp)
    3500:	c5 f9 7f 44 24 10    	vmovdqa %xmm0,0x10(%rsp)
    3506:	48 29 ce             	sub    %rcx,%rsi
    3509:	e8 42 ec ff ff       	call   2150 <_ZdlPvm@plt>
    350e:	4c 8b 44 24 20       	mov    0x20(%rsp),%r8
    3513:	c5 f9 6f 44 24 10    	vmovdqa 0x10(%rsp),%xmm0
    3519:	4d 01 e0             	add    %r12,%r8
    351c:	c5 fa 7f 03          	vmovdqu %xmm0,(%rbx)
    3520:	4c 89 43 10          	mov    %r8,0x10(%rbx)
    3524:	e9 92 fe ff ff       	jmp    33bb <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x11b>
    3529:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    3530:	48 8d 05 49 27 00 00 	lea    0x2749(%rip),%rax        # 5c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    3537:	48 8b 3c 24          	mov    (%rsp),%rdi
    353b:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    3540:	48 83 c0 28          	add    $0x28,%rax
    3544:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    354b:	00 
    354c:	48 8d 05 75 27 00 00 	lea    0x2775(%rip),%rax        # 5cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3553:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    3558:	e8 f3 ea ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
    355d:	48 8d bc 24 b8 00 00 	lea    0xb8(%rsp),%rdi
    3564:	00 
    3565:	e8 f6 ec ff ff       	call   2260 <_ZNSt12__basic_fileIcED1Ev@plt>
    356a:	48 8d 05 67 26 00 00 	lea    0x2667(%rip),%rax        # 5bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3571:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
    3578:	00 
    3579:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    357e:	e8 3d ec ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
    3583:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    3587:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    358c:	48 8b 0d 05 28 00 00 	mov    0x2805(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3593:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    3598:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    359d:	48 8d 05 14 26 00 00 	lea    0x2614(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    35a4:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    35ab:	00 
    35ac:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
    35b3:	00 00 
    35b5:	e8 d6 ea ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    35ba:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    35be:	48 89 d8             	mov    %rbx,%rax
    35c1:	5b                   	pop    %rbx
    35c2:	41 5c                	pop    %r12
    35c4:	41 5d                	pop    %r13
    35c6:	41 5e                	pop    %r14
    35c8:	41 5f                	pop    %r15
    35ca:	5d                   	pop    %rbp
    35cb:	c3                   	ret
    35cc:	0f 1f 40 00          	nopl   0x0(%rax)
    35d0:	8b 77 20             	mov    0x20(%rdi),%esi
    35d3:	83 ce 04             	or     $0x4,%esi
    35d6:	e8 45 ec ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    35db:	e9 c6 fd ff ff       	jmp    33a6 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x106>
    35e0:	49 bc f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%r12
    35e7:	ff ff 7f 
    35ea:	e9 7e fe ff ff       	jmp    346d <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1cd>
    35ef:	48 89 c3             	mov    %rax,%rbx
    35f2:	eb 11                	jmp    3605 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x365>
    35f4:	48 89 c3             	mov    %rax,%rbx
    35f7:	eb 28                	jmp    3621 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x381>
    35f9:	48 8b 3c 24          	mov    (%rsp),%rdi
    35fd:	c5 f8 77             	vzeroupper
    3600:	e8 db eb ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
    3605:	49 8b 47 e8          	mov    -0x18(%r15),%rax
    3609:	48 8b 0d 88 27 00 00 	mov    0x2788(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3610:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
    3615:	48 89 4c 04 40       	mov    %rcx,0x40(%rsp,%rax,1)
    361a:	31 c0                	xor    %eax,%eax
    361c:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
    3621:	48 8d 05 90 25 00 00 	lea    0x2590(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3628:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    362d:	48 89 84 24 40 01 00 	mov    %rax,0x140(%rsp)
    3634:	00 
    3635:	c5 f8 77             	vzeroupper
    3638:	e8 53 ea ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    363d:	48 89 df             	mov    %rbx,%rdi
    3640:	e8 eb eb ff ff       	call   2230 <_Unwind_Resume@plt>
    3645:	48 89 c3             	mov    %rax,%rbx
    3648:	eb af                	jmp    35f9 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x359>
    364a:	48 89 c7             	mov    %rax,%rdi
    364d:	c5 f8 77             	vzeroupper
    3650:	e8 4b ea ff ff       	call   20a0 <__cxa_begin_catch@plt>
    3655:	e8 b6 eb ff ff       	call   2210 <__cxa_end_catch@plt>
    365a:	e9 fe fe ff ff       	jmp    355d <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2bd>
    365f:	48 8d 3d ae 09 00 00 	lea    0x9ae(%rip),%rdi        # 4014 <_IO_stdin_used+0x14>
    3666:	e8 55 ea ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
    366b:	49 89 c4             	mov    %rax,%r12
    366e:	48 8b 3b             	mov    (%rbx),%rdi
    3671:	48 8b 73 10          	mov    0x10(%rbx),%rsi
    3675:	48 29 fe             	sub    %rdi,%rsi
    3678:	48 85 ff             	test   %rdi,%rdi
    367b:	74 18                	je     3695 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3f5>
    367d:	c5 f8 77             	vzeroupper
    3680:	e8 cb ea ff ff       	call   2150 <_ZdlPvm@plt>
    3685:	4c 89 ef             	mov    %r13,%rdi
    3688:	e8 e3 e9 ff ff       	call   2070 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@plt>
    368d:	4c 89 e7             	mov    %r12,%rdi
    3690:	e8 9b eb ff ff       	call   2230 <_Unwind_Resume@plt>
    3695:	c5 f8 77             	vzeroupper
    3698:	eb eb                	jmp    3685 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3e5>

Disassembly of section .fini:

000000000000369c <_fini>:
    369c:	48 83 ec 08          	sub    $0x8,%rsp
    36a0:	48 83 c4 08          	add    $0x8,%rsp
    36a4:	c3                   	ret
