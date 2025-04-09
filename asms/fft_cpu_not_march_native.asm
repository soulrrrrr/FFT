
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
    22c5:	48 8d 3d dc 1d 00 00 	lea    0x1ddc(%rip),%rdi        # 40a8 <_IO_stdin_used+0xa8>
    22cc:	e8 ef fd ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
    22d1:	48 89 ef             	mov    %rbp,%rdi
    22d4:	e8 b7 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    22d9:	48 89 df             	mov    %rbx,%rdi
    22dc:	e8 4f ff ff ff       	call   2230 <_Unwind_Resume@plt>
    22e1:	48 89 c3             	mov    %rax,%rbx
    22e4:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    22e9:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    22ee:	48 29 c6             	sub    %rax,%rsi
    22f1:	48 85 c0             	test   %rax,%rax
    22f4:	74 08                	je     22fe <main.cold+0x39>
    22f6:	48 89 c7             	mov    %rax,%rdi
    22f9:	e8 52 fe ff ff       	call   2150 <_ZdlPvm@plt>
    22fe:	48 89 df             	mov    %rbx,%rdi
    2301:	e8 2a ff ff ff       	call   2230 <_Unwind_Resume@plt>
    2306:	48 89 ef             	mov    %rbp,%rdi
    2309:	e8 c2 fe ff ff       	call   21d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    230e:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    2313:	e8 78 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2318:	4d 85 ed             	test   %r13,%r13
    231b:	74 c7                	je     22e4 <main.cold+0x1f>
    231d:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
    2322:	4c 89 ef             	mov    %r13,%rdi
    2325:	e8 26 fe ff ff       	call   2150 <_ZdlPvm@plt>
    232a:	eb b8                	jmp    22e4 <main.cold+0x1f>
    232c:	e8 6f fd ff ff       	call   20a0 <__cxa_begin_catch@plt>
    2331:	e8 da fe ff ff       	call   2210 <__cxa_end_catch@plt>
    2336:	e9 5f 06 00 00       	jmp    299a <main+0x5da>
    233b:	48 89 df             	mov    %rbx,%rdi
    233e:	48 89 eb             	mov    %rbp,%rbx
    2341:	e8 4a fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2346:	eb d0                	jmp    2318 <main.cold+0x53>
    2348:	48 89 ef             	mov    %rbp,%rdi
    234b:	e8 40 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2350:	48 89 df             	mov    %rbx,%rdi
    2353:	4c 89 e3             	mov    %r12,%rbx
    2356:	e8 35 fe ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    235b:	eb bb                	jmp    2318 <main.cold+0x53>
    235d:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    2362:	e8 79 fe ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
    2367:	48 8b 05 e2 38 00 00 	mov    0x38e2(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    236e:	48 8b 0d e3 38 00 00 	mov    0x38e3(%rip),%rcx        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2375:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
    237c:	00 
    237d:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2381:	48 89 8c 04 c0 00 00 	mov    %rcx,0xc0(%rsp,%rax,1)
    2388:	00 
    2389:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    238e:	48 8d 05 23 38 00 00 	lea    0x3823(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2395:	48 89 84 24 b8 01 00 	mov    %rax,0x1b8(%rsp)
    239c:	00 
    239d:	e8 ee fc ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    23a2:	e9 67 ff ff ff       	jmp    230e <main.cold+0x49>
    23a7:	48 8d 3d c2 1c 00 00 	lea    0x1cc2(%rip),%rdi        # 4070 <_IO_stdin_used+0x70>
    23ae:	e8 3d fd ff ff       	call   20f0 <_ZSt19__throw_logic_errorPKc@plt>
    23b3:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    23ba:	00 00 00 
    23bd:	0f 1f 00             	nopl   (%rax)

00000000000023c0 <main>:
    23c0:	41 57                	push   %r15
    23c2:	41 56                	push   %r14
    23c4:	41 55                	push   %r13
    23c6:	41 54                	push   %r12
    23c8:	55                   	push   %rbp
    23c9:	53                   	push   %rbx
    23ca:	48 81 ec c8 02 00 00 	sub    $0x2c8,%rsp
    23d1:	83 ff 01             	cmp    $0x1,%edi
    23d4:	0f 8e bc 06 00 00    	jle    2a96 <main+0x6d6>
    23da:	4c 8b 66 08          	mov    0x8(%rsi),%r12
    23de:	4c 8d ac 24 d0 00 00 	lea    0xd0(%rsp),%r13
    23e5:	00 
    23e6:	48 8d ac 24 c0 00 00 	lea    0xc0(%rsp),%rbp
    23ed:	00 
    23ee:	4c 89 ac 24 c0 00 00 	mov    %r13,0xc0(%rsp)
    23f5:	00 
    23f6:	4d 85 e4             	test   %r12,%r12
    23f9:	0f 84 a8 ff ff ff    	je     23a7 <main.cold+0xe2>
    23ff:	4c 89 e7             	mov    %r12,%rdi
    2402:	e8 a9 fc ff ff       	call   20b0 <strlen@plt>
    2407:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
    240e:	00 
    240f:	48 89 c3             	mov    %rax,%rbx
    2412:	48 83 f8 0f          	cmp    $0xf,%rax
    2416:	0f 87 29 06 00 00    	ja     2a45 <main+0x685>
    241c:	48 83 f8 01          	cmp    $0x1,%rax
    2420:	0f 85 9d 06 00 00    	jne    2ac3 <main+0x703>
    2426:	41 0f b6 04 24       	movzbl (%r12),%eax
    242b:	88 84 24 d0 00 00 00 	mov    %al,0xd0(%rsp)
    2432:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
    2439:	00 
    243a:	48 8b 94 24 c0 00 00 	mov    0xc0(%rsp),%rdx
    2441:	00 
    2442:	4c 8d 74 24 40       	lea    0x40(%rsp),%r14
    2447:	48 89 ee             	mov    %rbp,%rsi
    244a:	4c 89 f7             	mov    %r14,%rdi
    244d:	48 89 84 24 c8 00 00 	mov    %rax,0xc8(%rsp)
    2454:	00 
    2455:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
    2459:	e8 02 0e 00 00       	call   3260 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>
    245e:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
    2463:	48 89 ef             	mov    %rbp,%rdi
    2466:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    246b:	e8 20 fd ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2470:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    2475:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
    247a:	48 29 c8             	sub    %rcx,%rax
    247d:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    2482:	48 89 c3             	mov    %rax,%rbx
    2485:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    248a:	48 89 c1             	mov    %rax,%rcx
    248d:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    2492:	48 c1 fb 03          	sar    $0x3,%rbx
    2496:	48 b8 f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%rax
    249d:	ff ff 7f 
    24a0:	49 89 df             	mov    %rbx,%r15
    24a3:	48 39 c8             	cmp    %rcx,%rax
    24a6:	0f 82 19 fe ff ff    	jb     22c5 <main.cold>
    24ac:	48 c7 44 24 68 00 00 	movq   $0x0,0x68(%rsp)
    24b3:	00 00 
    24b5:	48 85 db             	test   %rbx,%rbx
    24b8:	0f 84 c4 05 00 00    	je     2a82 <main+0x6c2>
    24be:	4c 8b 64 24 18       	mov    0x18(%rsp),%r12
    24c3:	4c 89 e7             	mov    %r12,%rdi
    24c6:	e8 75 fc ff ff       	call   2140 <_Znwm@plt>
    24cb:	49 89 c5             	mov    %rax,%r13
    24ce:	4c 89 e0             	mov    %r12,%rax
    24d1:	48 83 f8 08          	cmp    $0x8,%rax
    24d5:	4f 8d 64 25 00       	lea    0x0(%r13,%r12,1),%r12
    24da:	4c 89 e8             	mov    %r13,%rax
    24dd:	4c 89 e6             	mov    %r12,%rsi
    24e0:	74 4f                	je     2531 <main+0x171>
    24e2:	48 89 d9             	mov    %rbx,%rcx
    24e5:	66 0f ef c0          	pxor   %xmm0,%xmm0
    24e9:	48 d1 e9             	shr    $1,%rcx
    24ec:	48 c1 e1 04          	shl    $0x4,%rcx
    24f0:	4a 8d 14 29          	lea    (%rcx,%r13,1),%rdx
    24f4:	80 e1 10             	and    $0x10,%cl
    24f7:	74 17                	je     2510 <main+0x150>
    24f9:	49 8d 45 10          	lea    0x10(%r13),%rax
    24fd:	41 0f 11 45 00       	movups %xmm0,0x0(%r13)
    2502:	48 39 c2             	cmp    %rax,%rdx
    2505:	74 19                	je     2520 <main+0x160>
    2507:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    250e:	00 00 
    2510:	0f 11 00             	movups %xmm0,(%rax)
    2513:	48 83 c0 20          	add    $0x20,%rax
    2517:	0f 11 40 f0          	movups %xmm0,-0x10(%rax)
    251b:	48 39 c2             	cmp    %rax,%rdx
    251e:	75 f0                	jne    2510 <main+0x150>
    2520:	48 89 d8             	mov    %rbx,%rax
    2523:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
    2527:	f6 c3 01             	test   $0x1,%bl
    252a:	74 0a                	je     2536 <main+0x176>
    252c:	49 8d 44 c5 00       	lea    0x0(%r13,%rax,8),%rax
    2531:	31 c9                	xor    %ecx,%ecx
    2533:	48 89 08             	mov    %rcx,(%rax)
    2536:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
    253b:	4c 89 64 24 68       	mov    %r12,0x68(%rsp)
    2540:	e8 fb fa ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    2545:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
    254a:	48 89 df             	mov    %rbx,%rdi
    254d:	48 8d 54 24 60       	lea    0x60(%rsp),%rdx
    2552:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    2557:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
    255c:	4c 89 f6             	mov    %r14,%rsi
    255f:	4c 89 6c 24 60       	mov    %r13,0x60(%rsp)
    2564:	e8 27 09 00 00       	call   2e90 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_>
    2569:	e8 d2 fa ff ff       	call   2040 <_ZNSt6chrono3_V212system_clock3nowEv@plt>
    256e:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    2573:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2577:	48 8d 3d 02 3c 00 00 	lea    0x3c02(%rip),%rdi        # 6180 <_ZSt4cout@GLIBCXX_3.4>
    257e:	48 29 c8             	sub    %rcx,%rax
    2581:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
    2586:	f3 0f 5e 05 82 1a 00 	divss  0x1a82(%rip),%xmm0        # 4010 <_IO_stdin_used+0x10>
    258d:	00 
    258e:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
    2592:	e8 69 fc ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2597:	48 89 c7             	mov    %rax,%rdi
    259a:	e8 a1 06 00 00       	call   2c40 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    259f:	48 83 7c 24 08 48    	cmpq   $0x48,0x8(%rsp)
    25a5:	b9 01 00 00 00       	mov    $0x1,%ecx
    25aa:	48 be 4b 59 86 38 d6 	movabs $0x346dc5d63886594b,%rsi
    25b1:	c5 6d 34 
    25b4:	77 3d                	ja     25f3 <main+0x233>
    25b6:	eb 44                	jmp    25fc <main+0x23c>
    25b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    25bf:	00 
    25c0:	48 81 fb e7 03 00 00 	cmp    $0x3e7,%rbx
    25c7:	0f 86 14 05 00 00    	jbe    2ae1 <main+0x721>
    25cd:	48 81 fb 0f 27 00 00 	cmp    $0x270f,%rbx
    25d4:	0f 86 0f 05 00 00    	jbe    2ae9 <main+0x729>
    25da:	48 89 d8             	mov    %rbx,%rax
    25dd:	83 c1 04             	add    $0x4,%ecx
    25e0:	48 f7 e6             	mul    %rsi
    25e3:	48 c1 ea 0b          	shr    $0xb,%rdx
    25e7:	48 81 fb 9f 86 01 00 	cmp    $0x1869f,%rbx
    25ee:	76 0c                	jbe    25fc <main+0x23c>
    25f0:	48 89 d3             	mov    %rdx,%rbx
    25f3:	48 83 fb 63          	cmp    $0x63,%rbx
    25f7:	77 c7                	ja     25c0 <main+0x200>
    25f9:	83 c1 01             	add    $0x1,%ecx
    25fc:	48 8d 9c 24 a0 00 00 	lea    0xa0(%rsp),%rbx
    2603:	00 
    2604:	41 89 ce             	mov    %ecx,%r14d
    2607:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
    260e:	00 
    260f:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
    2616:	00 00 00 00 00 
    261b:	4c 89 f6             	mov    %r14,%rsi
    261e:	48 89 df             	mov    %rbx,%rdi
    2621:	89 4c 24 20          	mov    %ecx,0x20(%rsp)
    2625:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
    262c:	00 
    262d:	c6 84 24 b0 00 00 00 	movb   $0x0,0xb0(%rsp)
    2634:	00 
    2635:	e8 b6 fb ff ff       	call   21f0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7reserveEm@plt>
    263a:	66 0f 6f 05 be 1a 00 	movdqa 0x1abe(%rip),%xmm0        # 4100 <_IO_stdin_used+0x100>
    2641:	00 
    2642:	8b 4c 24 20          	mov    0x20(%rsp),%ecx
    2646:	48 8b bc 24 a0 00 00 	mov    0xa0(%rsp),%rdi
    264d:	00 
    264e:	0f 29 84 24 c0 00 00 	movaps %xmm0,0xc0(%rsp)
    2655:	00 
    2656:	66 0f 6f 05 b2 1a 00 	movdqa 0x1ab2(%rip),%xmm0        # 4110 <_IO_stdin_used+0x110>
    265d:	00 
    265e:	83 e9 01             	sub    $0x1,%ecx
    2661:	48 81 7c 24 08 18 03 	cmpq   $0x318,0x8(%rsp)
    2668:	00 00 
    266a:	0f 29 84 24 d0 00 00 	movaps %xmm0,0xd0(%rsp)
    2671:	00 
    2672:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4120 <_IO_stdin_used+0x120>
    2679:	00 
    267a:	0f 29 84 24 e0 00 00 	movaps %xmm0,0xe0(%rsp)
    2681:	00 
    2682:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4130 <_IO_stdin_used+0x130>
    2689:	00 
    268a:	0f 29 84 24 f0 00 00 	movaps %xmm0,0xf0(%rsp)
    2691:	00 
    2692:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4140 <_IO_stdin_used+0x140>
    2699:	00 
    269a:	0f 29 84 24 00 01 00 	movaps %xmm0,0x100(%rsp)
    26a1:	00 
    26a2:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4150 <_IO_stdin_used+0x150>
    26a9:	00 
    26aa:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
    26b1:	00 
    26b2:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4160 <_IO_stdin_used+0x160>
    26b9:	00 
    26ba:	0f 29 84 24 20 01 00 	movaps %xmm0,0x120(%rsp)
    26c1:	00 
    26c2:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4170 <_IO_stdin_used+0x170>
    26c9:	00 
    26ca:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
    26d1:	00 
    26d2:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4180 <_IO_stdin_used+0x180>
    26d9:	00 
    26da:	0f 29 84 24 40 01 00 	movaps %xmm0,0x140(%rsp)
    26e1:	00 
    26e2:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 4190 <_IO_stdin_used+0x190>
    26e9:	00 
    26ea:	0f 29 84 24 50 01 00 	movaps %xmm0,0x150(%rsp)
    26f1:	00 
    26f2:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 41a0 <_IO_stdin_used+0x1a0>
    26f9:	00 
    26fa:	0f 29 84 24 60 01 00 	movaps %xmm0,0x160(%rsp)
    2701:	00 
    2702:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 41b0 <_IO_stdin_used+0x1b0>
    2709:	00 
    270a:	0f 29 84 24 70 01 00 	movaps %xmm0,0x170(%rsp)
    2711:	00 
    2712:	66 0f 6f 05 a6 1a 00 	movdqa 0x1aa6(%rip),%xmm0        # 41c0 <_IO_stdin_used+0x1c0>
    2719:	00 
    271a:	0f 11 84 24 79 01 00 	movups %xmm0,0x179(%rsp)
    2721:	00 
    2722:	76 6a                	jbe    278e <main+0x3ce>
    2724:	48 be c3 f5 28 5c 8f 	movabs $0x28f5c28f5c28f5c3,%rsi
    272b:	c2 f5 28 
    272e:	66 90                	xchg   %ax,%ax
    2730:	4c 89 fa             	mov    %r15,%rdx
    2733:	48 c1 ea 02          	shr    $0x2,%rdx
    2737:	48 89 d0             	mov    %rdx,%rax
    273a:	48 f7 e6             	mul    %rsi
    273d:	4c 89 f8             	mov    %r15,%rax
    2740:	49 89 d0             	mov    %rdx,%r8
    2743:	48 83 e2 fc          	and    $0xfffffffffffffffc,%rdx
    2747:	49 c1 e8 02          	shr    $0x2,%r8
    274b:	4c 01 c2             	add    %r8,%rdx
    274e:	48 8d 14 92          	lea    (%rdx,%rdx,4),%rdx
    2752:	48 c1 e2 02          	shl    $0x2,%rdx
    2756:	48 29 d0             	sub    %rdx,%rax
    2759:	4c 89 fa             	mov    %r15,%rdx
    275c:	4d 89 c7             	mov    %r8,%r15
    275f:	41 89 c8             	mov    %ecx,%r8d
    2762:	48 01 c0             	add    %rax,%rax
    2765:	44 0f b6 8c 04 c1 00 	movzbl 0xc1(%rsp,%rax,1),%r9d
    276c:	00 00 
    276e:	46 88 0c 07          	mov    %r9b,(%rdi,%r8,1)
    2772:	0f b6 84 04 c0 00 00 	movzbl 0xc0(%rsp,%rax,1),%eax
    2779:	00 
    277a:	44 8d 41 ff          	lea    -0x1(%rcx),%r8d
    277e:	83 e9 02             	sub    $0x2,%ecx
    2781:	42 88 04 07          	mov    %al,(%rdi,%r8,1)
    2785:	48 81 fa 0f 27 00 00 	cmp    $0x270f,%rdx
    278c:	77 a2                	ja     2730 <main+0x370>
    278e:	41 8d 47 30          	lea    0x30(%r15),%eax
    2792:	49 83 ff 09          	cmp    $0x9,%r15
    2796:	76 18                	jbe    27b0 <main+0x3f0>
    2798:	4d 01 ff             	add    %r15,%r15
    279b:	42 0f b6 84 3c c1 00 	movzbl 0xc1(%rsp,%r15,1),%eax
    27a2:	00 00 
    27a4:	88 47 01             	mov    %al,0x1(%rdi)
    27a7:	42 0f b6 84 3c c0 00 	movzbl 0xc0(%rsp,%r15,1),%eax
    27ae:	00 00 
    27b0:	88 07                	mov    %al,(%rdi)
    27b2:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
    27b9:	00 
    27ba:	31 f6                	xor    %esi,%esi
    27bc:	48 89 df             	mov    %rbx,%rdi
    27bf:	4c 89 b4 24 a8 00 00 	mov    %r14,0xa8(%rsp)
    27c6:	00 
    27c7:	48 8d 15 7e 18 00 00 	lea    0x187e(%rip),%rdx        # 404c <_IO_stdin_used+0x4c>
    27ce:	42 c6 04 30 00       	movb   $0x0,(%rax,%r14,1)
    27d3:	e8 88 f8 ff ff       	call   2060 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6insertEmPKc@plt>
    27d8:	48 89 c6             	mov    %rax,%rsi
    27db:	48 89 ef             	mov    %rbp,%rdi
    27de:	e8 ed f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
    27e3:	48 8d 35 77 18 00 00 	lea    0x1877(%rip),%rsi        # 4061 <_IO_stdin_used+0x61>
    27ea:	48 89 ef             	mov    %rbp,%rdi
    27ed:	e8 8e fa ff ff       	call   2280 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc@plt>
    27f2:	48 89 c6             	mov    %rax,%rsi
    27f5:	48 8d 84 24 80 00 00 	lea    0x80(%rsp),%rax
    27fc:	00 
    27fd:	4c 8d b4 24 b8 01 00 	lea    0x1b8(%rsp),%r14
    2804:	00 
    2805:	48 89 c7             	mov    %rax,%rdi
    2808:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    280d:	e8 be f8 ff ff       	call   20d0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EOS4_@plt>
    2812:	48 89 ef             	mov    %rbp,%rdi
    2815:	e8 76 f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    281a:	48 89 df             	mov    %rbx,%rdi
    281d:	e8 6e f9 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2822:	4c 89 f7             	mov    %r14,%rdi
    2825:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    282a:	e8 51 f8 ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
    282f:	48 8d 05 82 33 00 00 	lea    0x3382(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2836:	66 0f ef c0          	pxor   %xmm0,%xmm0
    283a:	31 f6                	xor    %esi,%esi
    283c:	48 89 84 24 b8 01 00 	mov    %rax,0x1b8(%rsp)
    2843:	00 
    2844:	31 c0                	xor    %eax,%eax
    2846:	66 89 84 24 98 02 00 	mov    %ax,0x298(%rsp)
    284d:	00 
    284e:	48 8b 05 fb 33 00 00 	mov    0x33fb(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    2855:	0f 29 84 24 a0 02 00 	movaps %xmm0,0x2a0(%rsp)
    285c:	00 
    285d:	0f 29 84 24 b0 02 00 	movaps %xmm0,0x2b0(%rsp)
    2864:	00 
    2865:	48 8b 78 e8          	mov    -0x18(%rax),%rdi
    2869:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
    2870:	00 
    2871:	48 8b 05 e0 33 00 00 	mov    0x33e0(%rip),%rax        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2878:	48 c7 84 24 90 02 00 	movq   $0x0,0x290(%rsp)
    287f:	00 00 00 00 00 
    2884:	48 01 ef             	add    %rbp,%rdi
    2887:	48 89 07             	mov    %rax,(%rdi)
    288a:	e8 21 f9 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    288f:	48 8d 05 ba 34 00 00 	lea    0x34ba(%rip),%rax        # 5d50 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    2896:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
    289d:	00 
    289e:	48 83 c0 28          	add    $0x28,%rax
    28a2:	48 89 84 24 b8 01 00 	mov    %rax,0x1b8(%rsp)
    28a9:	00 
    28aa:	48 8d 84 24 c8 00 00 	lea    0xc8(%rsp),%rax
    28b1:	00 
    28b2:	48 89 c7             	mov    %rax,%rdi
    28b5:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    28ba:	48 89 c3             	mov    %rax,%rbx
    28bd:	e8 9e f8 ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
    28c2:	48 89 de             	mov    %rbx,%rsi
    28c5:	4c 89 f7             	mov    %r14,%rdi
    28c8:	e8 e3 f8 ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    28cd:	48 8b b4 24 80 00 00 	mov    0x80(%rsp),%rsi
    28d4:	00 
    28d5:	ba 10 00 00 00       	mov    $0x10,%edx
    28da:	48 89 df             	mov    %rbx,%rdi
    28dd:	e8 4e f8 ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
    28e2:	48 8b 94 24 c0 00 00 	mov    0xc0(%rsp),%rdx
    28e9:	00 
    28ea:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    28ee:	48 01 ef             	add    %rbp,%rdi
    28f1:	48 85 c0             	test   %rax,%rax
    28f4:	0f 84 d7 01 00 00    	je     2ad1 <main+0x711>
    28fa:	31 f6                	xor    %esi,%esi
    28fc:	e8 1f f9 ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    2901:	4c 89 eb             	mov    %r13,%rbx
    2904:	4c 8d 3d 5d 17 00 00 	lea    0x175d(%rip),%r15        # 4068 <_IO_stdin_used+0x68>
    290b:	4d 39 ec             	cmp    %r13,%r12
    290e:	74 51                	je     2961 <main+0x5a1>
    2910:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2914:	48 89 ef             	mov    %rbp,%rdi
    2917:	f3 0f 5a 03          	cvtss2sd (%rbx),%xmm0
    291b:	e8 e0 f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2920:	ba 01 00 00 00       	mov    $0x1,%edx
    2925:	48 8d 35 3a 17 00 00 	lea    0x173a(%rip),%rsi        # 4066 <_IO_stdin_used+0x66>
    292c:	48 89 c7             	mov    %rax,%rdi
    292f:	49 89 c6             	mov    %rax,%r14
    2932:	e8 39 f8 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2937:	66 0f ef c0          	pxor   %xmm0,%xmm0
    293b:	4c 89 f7             	mov    %r14,%rdi
    293e:	f3 0f 5a 43 04       	cvtss2sd 0x4(%rbx),%xmm0
    2943:	e8 b8 f8 ff ff       	call   2200 <_ZNSo9_M_insertIdEERSoT_@plt>
    2948:	48 89 c7             	mov    %rax,%rdi
    294b:	ba 01 00 00 00       	mov    $0x1,%edx
    2950:	4c 89 fe             	mov    %r15,%rsi
    2953:	e8 18 f8 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2958:	48 83 c3 08          	add    $0x8,%rbx
    295c:	49 39 dc             	cmp    %rbx,%r12
    295f:	75 af                	jne    2910 <main+0x550>
    2961:	48 8d 05 10 34 00 00 	lea    0x3410(%rip),%rax        # 5d78 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x40>
    2968:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    296d:	f3 0f 7e 05 33 34 00 	movq   0x3433(%rip),%xmm0        # 5da8 <_ZTVSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x70>
    2974:	00 
    2975:	48 89 84 24 b8 01 00 	mov    %rax,0x1b8(%rsp)
    297c:	00 
    297d:	48 8d 05 44 33 00 00 	lea    0x3344(%rip),%rax        # 5cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    2984:	66 48 0f 6e c8       	movq   %rax,%xmm1
    2989:	66 0f 6c c1          	punpcklqdq %xmm1,%xmm0
    298d:	0f 29 84 24 c0 00 00 	movaps %xmm0,0xc0(%rsp)
    2994:	00 
    2995:	e8 b6 f6 ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
    299a:	48 8d bc 24 30 01 00 	lea    0x130(%rsp),%rdi
    29a1:	00 
    29a2:	e8 b9 f8 ff ff       	call   2260 <_ZNSt12__basic_fileIcED1Ev@plt>
    29a7:	48 8d 05 2a 32 00 00 	lea    0x322a(%rip),%rax        # 5bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29ae:	48 8d bc 24 00 01 00 	lea    0x100(%rsp),%rdi
    29b5:	00 
    29b6:	48 89 84 24 c8 00 00 	mov    %rax,0xc8(%rsp)
    29bd:	00 
    29be:	e8 fd f7 ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
    29c3:	48 8b 05 86 32 00 00 	mov    0x3286(%rip),%rax        # 5c50 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    29ca:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    29cf:	48 8b 0d 82 32 00 00 	mov    0x3282(%rip),%rcx        # 5c58 <_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29d6:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
    29dd:	00 
    29de:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    29e2:	48 89 8c 04 c0 00 00 	mov    %rcx,0xc0(%rsp,%rax,1)
    29e9:	00 
    29ea:	48 8d 05 c7 31 00 00 	lea    0x31c7(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    29f1:	48 89 84 24 b8 01 00 	mov    %rax,0x1b8(%rsp)
    29f8:	00 
    29f9:	e8 92 f6 ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    29fe:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    2a03:	e8 88 f7 ff ff       	call   2190 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    2a08:	4d 85 ed             	test   %r13,%r13
    2a0b:	74 0d                	je     2a1a <main+0x65a>
    2a0d:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    2a12:	4c 89 ef             	mov    %r13,%rdi
    2a15:	e8 36 f7 ff ff       	call   2150 <_ZdlPvm@plt>
    2a1a:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    2a1f:	48 85 ff             	test   %rdi,%rdi
    2a22:	74 0d                	je     2a31 <main+0x671>
    2a24:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    2a29:	48 29 fe             	sub    %rdi,%rsi
    2a2c:	e8 1f f7 ff ff       	call   2150 <_ZdlPvm@plt>
    2a31:	31 c0                	xor    %eax,%eax
    2a33:	48 81 c4 c8 02 00 00 	add    $0x2c8,%rsp
    2a3a:	5b                   	pop    %rbx
    2a3b:	5d                   	pop    %rbp
    2a3c:	41 5c                	pop    %r12
    2a3e:	41 5d                	pop    %r13
    2a40:	41 5e                	pop    %r14
    2a42:	41 5f                	pop    %r15
    2a44:	c3                   	ret
    2a45:	48 89 ef             	mov    %rbp,%rdi
    2a48:	48 8d b4 24 a0 00 00 	lea    0xa0(%rsp),%rsi
    2a4f:	00 
    2a50:	31 d2                	xor    %edx,%edx
    2a52:	e8 f9 f7 ff ff       	call   2250 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@plt>
    2a57:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
    2a5e:	00 
    2a5f:	48 89 c7             	mov    %rax,%rdi
    2a62:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
    2a69:	00 
    2a6a:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
    2a71:	00 
    2a72:	48 89 da             	mov    %rbx,%rdx
    2a75:	4c 89 e6             	mov    %r12,%rsi
    2a78:	e8 93 f6 ff ff       	call   2110 <memcpy@plt>
    2a7d:	e9 b0 f9 ff ff       	jmp    2432 <main+0x72>
    2a82:	31 d2                	xor    %edx,%edx
    2a84:	31 f6                	xor    %esi,%esi
    2a86:	45 31 ed             	xor    %r13d,%r13d
    2a89:	45 31 e4             	xor    %r12d,%r12d
    2a8c:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
    2a91:	e9 a0 fa ff ff       	jmp    2536 <main+0x176>
    2a96:	48 8d 1d 03 38 00 00 	lea    0x3803(%rip),%rbx        # 62a0 <_ZSt4cerr@GLIBCXX_3.4>
    2a9d:	ba 1d 00 00 00       	mov    $0x1d,%edx
    2aa2:	48 8d 35 85 15 00 00 	lea    0x1585(%rip),%rsi        # 402e <_IO_stdin_used+0x2e>
    2aa9:	48 89 df             	mov    %rbx,%rdi
    2aac:	e8 bf f6 ff ff       	call   2170 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2ab1:	48 89 df             	mov    %rbx,%rdi
    2ab4:	e8 87 01 00 00       	call   2c40 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    2ab9:	b8 01 00 00 00       	mov    $0x1,%eax
    2abe:	e9 70 ff ff ff       	jmp    2a33 <main+0x673>
    2ac3:	48 85 c0             	test   %rax,%rax
    2ac6:	0f 84 66 f9 ff ff    	je     2432 <main+0x72>
    2acc:	4c 89 ef             	mov    %r13,%rdi
    2acf:	eb a1                	jmp    2a72 <main+0x6b2>
    2ad1:	8b 77 20             	mov    0x20(%rdi),%esi
    2ad4:	83 ce 04             	or     $0x4,%esi
    2ad7:	e8 44 f7 ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    2adc:	e9 20 fe ff ff       	jmp    2901 <main+0x541>
    2ae1:	83 c1 02             	add    $0x2,%ecx
    2ae4:	e9 13 fb ff ff       	jmp    25fc <main+0x23c>
    2ae9:	83 c1 03             	add    $0x3,%ecx
    2aec:	e9 0b fb ff ff       	jmp    25fc <main+0x23c>
    2af1:	48 89 c3             	mov    %rax,%rbx
    2af4:	e9 d8 f7 ff ff       	jmp    22d1 <main.cold+0xc>
    2af9:	48 89 c3             	mov    %rax,%rbx
    2afc:	e9 17 f8 ff ff       	jmp    2318 <main.cold+0x53>
    2b01:	e9 db f7 ff ff       	jmp    22e1 <main.cold+0x1c>
    2b06:	48 89 c3             	mov    %rax,%rbx
    2b09:	e9 f8 f7 ff ff       	jmp    2306 <main.cold+0x41>
    2b0e:	48 89 c7             	mov    %rax,%rdi
    2b11:	e9 16 f8 ff ff       	jmp    232c <main.cold+0x67>
    2b16:	48 89 c5             	mov    %rax,%rbp
    2b19:	e9 1d f8 ff ff       	jmp    233b <main.cold+0x76>
    2b1e:	49 89 c4             	mov    %rax,%r12
    2b21:	e9 2a f8 ff ff       	jmp    2350 <main.cold+0x8b>
    2b26:	49 89 c4             	mov    %rax,%r12
    2b29:	e9 1a f8 ff ff       	jmp    2348 <main.cold+0x83>
    2b2e:	48 89 c3             	mov    %rax,%rbx
    2b31:	e9 53 f8 ff ff       	jmp    2389 <main.cold+0xc4>
    2b36:	48 89 c3             	mov    %rax,%rbx
    2b39:	e9 29 f8 ff ff       	jmp    2367 <main.cold+0xa2>
    2b3e:	48 89 c3             	mov    %rax,%rbx
    2b41:	e9 17 f8 ff ff       	jmp    235d <main.cold+0x98>
    2b46:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2b4d:	00 00 00 

0000000000002b50 <_start>:
    2b50:	31 ed                	xor    %ebp,%ebp
    2b52:	49 89 d1             	mov    %rdx,%r9
    2b55:	5e                   	pop    %rsi
    2b56:	48 89 e2             	mov    %rsp,%rdx
    2b59:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    2b5d:	50                   	push   %rax
    2b5e:	54                   	push   %rsp
    2b5f:	45 31 c0             	xor    %r8d,%r8d
    2b62:	31 c9                	xor    %ecx,%ecx
    2b64:	48 8d 3d 55 f8 ff ff 	lea    -0x7ab(%rip),%rdi        # 23c0 <main>
    2b6b:	ff 15 57 34 00 00    	call   *0x3457(%rip)        # 5fc8 <__libc_start_main@GLIBC_2.34>
    2b71:	f4                   	hlt
    2b72:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2b79:	00 00 00 
    2b7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002b80 <deregister_tm_clones>:
    2b80:	48 8d 3d c9 35 00 00 	lea    0x35c9(%rip),%rdi        # 6150 <__TMC_END__>
    2b87:	48 8d 05 c2 35 00 00 	lea    0x35c2(%rip),%rax        # 6150 <__TMC_END__>
    2b8e:	48 39 f8             	cmp    %rdi,%rax
    2b91:	74 15                	je     2ba8 <deregister_tm_clones+0x28>
    2b93:	48 8b 05 36 34 00 00 	mov    0x3436(%rip),%rax        # 5fd0 <_ITM_deregisterTMCloneTable@Base>
    2b9a:	48 85 c0             	test   %rax,%rax
    2b9d:	74 09                	je     2ba8 <deregister_tm_clones+0x28>
    2b9f:	ff e0                	jmp    *%rax
    2ba1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2ba8:	c3                   	ret
    2ba9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002bb0 <register_tm_clones>:
    2bb0:	48 8d 3d 99 35 00 00 	lea    0x3599(%rip),%rdi        # 6150 <__TMC_END__>
    2bb7:	48 8d 35 92 35 00 00 	lea    0x3592(%rip),%rsi        # 6150 <__TMC_END__>
    2bbe:	48 29 fe             	sub    %rdi,%rsi
    2bc1:	48 89 f0             	mov    %rsi,%rax
    2bc4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    2bc8:	48 c1 f8 03          	sar    $0x3,%rax
    2bcc:	48 01 c6             	add    %rax,%rsi
    2bcf:	48 d1 fe             	sar    $1,%rsi
    2bd2:	74 14                	je     2be8 <register_tm_clones+0x38>
    2bd4:	48 8b 05 05 34 00 00 	mov    0x3405(%rip),%rax        # 5fe0 <_ITM_registerTMCloneTable@Base>
    2bdb:	48 85 c0             	test   %rax,%rax
    2bde:	74 08                	je     2be8 <register_tm_clones+0x38>
    2be0:	ff e0                	jmp    *%rax
    2be2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2be8:	c3                   	ret
    2be9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002bf0 <__do_global_dtors_aux>:
    2bf0:	f3 0f 1e fa          	endbr64
    2bf4:	80 3d b5 37 00 00 00 	cmpb   $0x0,0x37b5(%rip)        # 63b0 <completed.0>
    2bfb:	75 2b                	jne    2c28 <__do_global_dtors_aux+0x38>
    2bfd:	55                   	push   %rbp
    2bfe:	48 83 3d ba 33 00 00 	cmpq   $0x0,0x33ba(%rip)        # 5fc0 <__cxa_finalize@GLIBC_2.2.5>
    2c05:	00 
    2c06:	48 89 e5             	mov    %rsp,%rbp
    2c09:	74 0c                	je     2c17 <__do_global_dtors_aux+0x27>
    2c0b:	48 8b 3d 2e 35 00 00 	mov    0x352e(%rip),%rdi        # 6140 <__dso_handle>
    2c12:	e8 89 f6 ff ff       	call   22a0 <__cxa_finalize@plt>
    2c17:	e8 64 ff ff ff       	call   2b80 <deregister_tm_clones>
    2c1c:	c6 05 8d 37 00 00 01 	movb   $0x1,0x378d(%rip)        # 63b0 <completed.0>
    2c23:	5d                   	pop    %rbp
    2c24:	c3                   	ret
    2c25:	0f 1f 00             	nopl   (%rax)
    2c28:	c3                   	ret
    2c29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002c30 <frame_dummy>:
    2c30:	f3 0f 1e fa          	endbr64
    2c34:	e9 77 ff ff ff       	jmp    2bb0 <register_tm_clones>
    2c39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002c40 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>:
    2c40:	55                   	push   %rbp
    2c41:	53                   	push   %rbx
    2c42:	48 83 ec 08          	sub    $0x8,%rsp
    2c46:	48 8b 07             	mov    (%rdi),%rax
    2c49:	48 8b 40 e8          	mov    -0x18(%rax),%rax
    2c4d:	48 8b ac 07 f0 00 00 	mov    0xf0(%rdi,%rax,1),%rbp
    2c54:	00 
    2c55:	48 85 ed             	test   %rbp,%rbp
    2c58:	0f 84 62 f6 ff ff    	je     22c0 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0.cold>
    2c5e:	80 7d 38 00          	cmpb   $0x0,0x38(%rbp)
    2c62:	48 89 fb             	mov    %rdi,%rbx
    2c65:	74 1a                	je     2c81 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x41>
    2c67:	0f be 75 43          	movsbl 0x43(%rbp),%esi
    2c6b:	48 89 df             	mov    %rbx,%rdi
    2c6e:	e8 bd f3 ff ff       	call   2030 <_ZNSo3putEc@plt>
    2c73:	48 83 c4 08          	add    $0x8,%rsp
    2c77:	5b                   	pop    %rbx
    2c78:	48 89 c7             	mov    %rax,%rdi
    2c7b:	5d                   	pop    %rbp
    2c7c:	e9 5f f4 ff ff       	jmp    20e0 <_ZNSo5flushEv@plt>
    2c81:	48 89 ef             	mov    %rbp,%rdi
    2c84:	e8 f7 f4 ff ff       	call   2180 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
    2c89:	48 8b 45 00          	mov    0x0(%rbp),%rax
    2c8d:	be 0a 00 00 00       	mov    $0xa,%esi
    2c92:	48 89 ef             	mov    %rbp,%rdi
    2c95:	ff 50 30             	call   *0x30(%rax)
    2c98:	0f be f0             	movsbl %al,%esi
    2c9b:	eb ce                	jmp    2c6b <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0+0x2b>
    2c9d:	0f 1f 00             	nopl   (%rax)

0000000000002ca0 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_>:
    2ca0:	41 56                	push   %r14
    2ca2:	41 55                	push   %r13
    2ca4:	41 54                	push   %r12
    2ca6:	55                   	push   %rbp
    2ca7:	53                   	push   %rbx
    2ca8:	48 83 ec 20          	sub    $0x20,%rsp
    2cac:	48 85 ff             	test   %rdi,%rdi
    2caf:	0f 84 11 01 00 00    	je     2dc6 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x126>
    2cb5:	4c 8b 36             	mov    (%rsi),%r14
    2cb8:	4c 8b 22             	mov    (%rdx),%r12
    2cbb:	48 89 fb             	mov    %rdi,%rbx
    2cbe:	0f 88 0f 01 00 00    	js     2dd3 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x133>
    2cc4:	66 0f ef ed          	pxor   %xmm5,%xmm5
    2cc8:	f2 48 0f 2a ef       	cvtsi2sd %rdi,%xmm5
    2ccd:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
    2cd3:	31 ed                	xor    %ebp,%ebp
    2cd5:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2cdc:	00 00 00 00 
    2ce0:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2ce4:	66 0f ef e4          	pxor   %xmm4,%xmm4
    2ce8:	45 31 ed             	xor    %r13d,%r13d
    2ceb:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
    2cef:	f2 0f 59 05 e9 13 00 	mulsd  0x13e9(%rip),%xmm0        # 40e0 <_IO_stdin_used+0xe0>
    2cf6:	00 
    2cf7:	0f 28 ec             	movaps %xmm4,%xmm5
    2cfa:	f2 0f 11 44 24 10    	movsd  %xmm0,0x10(%rsp)
    2d00:	66 0f ef c9          	pxor   %xmm1,%xmm1
    2d04:	f3 0f 11 6c 24 0c    	movss  %xmm5,0xc(%rsp)
    2d0a:	f2 41 0f 2a cd       	cvtsi2sd %r13d,%xmm1
    2d0f:	f2 0f 59 4c 24 10    	mulsd  0x10(%rsp),%xmm1
    2d15:	f3 0f 11 64 24 08    	movss  %xmm4,0x8(%rsp)
    2d1b:	f2 0f 5e 4c 24 18    	divsd  0x18(%rsp),%xmm1
    2d21:	f2 0f 5a c9          	cvtsd2ss %xmm1,%xmm1
    2d25:	0f 28 c1             	movaps %xmm1,%xmm0
    2d28:	0f 57 05 c1 13 00 00 	xorps  0x13c1(%rip),%xmm0        # 40f0 <_IO_stdin_used+0xf0>
    2d2f:	f3 0f 11 4c 24 04    	movss  %xmm1,0x4(%rsp)
    2d35:	e8 e6 f3 ff ff       	call   2120 <sinf@plt>
    2d3a:	f3 0f 10 4c 24 04    	movss  0x4(%rsp),%xmm1
    2d40:	f3 0f 11 04 24       	movss  %xmm0,(%rsp)
    2d45:	0f 28 c1             	movaps %xmm1,%xmm0
    2d48:	e8 b3 f3 ff ff       	call   2100 <cosf@plt>
    2d4d:	f3 43 0f 10 3c ee    	movss  (%r14,%r13,8),%xmm7
    2d53:	f3 0f 10 1c 24       	movss  (%rsp),%xmm3
    2d58:	f3 43 0f 10 4c ee 04 	movss  0x4(%r14,%r13,8),%xmm1
    2d5f:	f3 0f 10 64 24 08    	movss  0x8(%rsp),%xmm4
    2d65:	0f 28 f7             	movaps %xmm7,%xmm6
    2d68:	44 0f 28 c7          	movaps %xmm7,%xmm8
    2d6c:	f3 0f 10 6c 24 0c    	movss  0xc(%rsp),%xmm5
    2d72:	f3 0f 59 f0          	mulss  %xmm0,%xmm6
    2d76:	0f 28 d1             	movaps %xmm1,%xmm2
    2d79:	f3 0f 59 d3          	mulss  %xmm3,%xmm2
    2d7d:	f3 44 0f 59 c3       	mulss  %xmm3,%xmm8
    2d82:	f3 0f 5c f2          	subss  %xmm2,%xmm6
    2d86:	0f 28 d1             	movaps %xmm1,%xmm2
    2d89:	f3 0f 59 d0          	mulss  %xmm0,%xmm2
    2d8d:	f3 41 0f 58 d0       	addss  %xmm8,%xmm2
    2d92:	0f 2e d6             	ucomiss %xmm6,%xmm2
    2d95:	7a 63                	jp     2dfa <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x15a>
    2d97:	49 83 c5 01          	add    $0x1,%r13
    2d9b:	f3 0f 58 ee          	addss  %xmm6,%xmm5
    2d9f:	f3 0f 58 e2          	addss  %xmm2,%xmm4
    2da3:	4c 39 eb             	cmp    %r13,%rbx
    2da6:	0f 85 54 ff ff ff    	jne    2d00 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x60>
    2dac:	f3 41 0f 11 2c ec    	movss  %xmm5,(%r12,%rbp,8)
    2db2:	f3 41 0f 11 64 ec 04 	movss  %xmm4,0x4(%r12,%rbp,8)
    2db9:	48 83 c5 01          	add    $0x1,%rbp
    2dbd:	48 39 eb             	cmp    %rbp,%rbx
    2dc0:	0f 85 1a ff ff ff    	jne    2ce0 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
    2dc6:	48 83 c4 20          	add    $0x20,%rsp
    2dca:	5b                   	pop    %rbx
    2dcb:	5d                   	pop    %rbp
    2dcc:	41 5c                	pop    %r12
    2dce:	41 5d                	pop    %r13
    2dd0:	41 5e                	pop    %r14
    2dd2:	c3                   	ret
    2dd3:	48 89 f8             	mov    %rdi,%rax
    2dd6:	48 89 fa             	mov    %rdi,%rdx
    2dd9:	66 0f ef c0          	pxor   %xmm0,%xmm0
    2ddd:	48 d1 e8             	shr    $1,%rax
    2de0:	83 e2 01             	and    $0x1,%edx
    2de3:	48 09 d0             	or     %rdx,%rax
    2de6:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    2deb:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
    2def:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
    2df5:	e9 d9 fe ff ff       	jmp    2cd3 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x33>
    2dfa:	0f 28 d0             	movaps %xmm0,%xmm2
    2dfd:	0f 28 c7             	movaps %xmm7,%xmm0
    2e00:	f3 0f 11 24 24       	movss  %xmm4,(%rsp)
    2e05:	49 83 c5 01          	add    $0x1,%r13
    2e09:	f3 0f 11 6c 24 04    	movss  %xmm5,0x4(%rsp)
    2e0f:	e8 5c f4 ff ff       	call   2270 <__mulsc3@plt>
    2e14:	f3 0f 10 6c 24 04    	movss  0x4(%rsp),%xmm5
    2e1a:	f3 0f 10 24 24       	movss  (%rsp),%xmm4
    2e1f:	66 48 0f 7e c0       	movq   %xmm0,%rax
    2e24:	66 0f 6f c8          	movdqa %xmm0,%xmm1
    2e28:	48 c1 e8 20          	shr    $0x20,%rax
    2e2c:	f3 0f 58 e9          	addss  %xmm1,%xmm5
    2e30:	66 0f 6e c0          	movd   %eax,%xmm0
    2e34:	f3 0f 58 e0          	addss  %xmm0,%xmm4
    2e38:	4c 39 eb             	cmp    %r13,%rbx
    2e3b:	0f 85 bf fe ff ff    	jne    2d00 <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x60>
    2e41:	e9 66 ff ff ff       	jmp    2dac <_Z9fft_naivemRSt6vectorISt7complexIfESaIS1_EES4_+0x10c>
    2e46:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2e4d:	00 00 00 

0000000000002e50 <_Z11bit_reversemi>:
    2e50:	85 f6                	test   %esi,%esi
    2e52:	7e 2c                	jle    2e80 <_Z11bit_reversemi+0x30>
    2e54:	31 d2                	xor    %edx,%edx
    2e56:	31 c0                	xor    %eax,%eax
    2e58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2e5f:	00 
    2e60:	48 89 f9             	mov    %rdi,%rcx
    2e63:	48 01 c0             	add    %rax,%rax
    2e66:	83 c2 01             	add    $0x1,%edx
    2e69:	48 d1 ef             	shr    $1,%rdi
    2e6c:	83 e1 01             	and    $0x1,%ecx
    2e6f:	48 09 c8             	or     %rcx,%rax
    2e72:	39 d6                	cmp    %edx,%esi
    2e74:	75 ea                	jne    2e60 <_Z11bit_reversemi+0x10>
    2e76:	c3                   	ret
    2e77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2e7e:	00 00 
    2e80:	31 c0                	xor    %eax,%eax
    2e82:	c3                   	ret
    2e83:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2e8a:	00 00 00 00 
    2e8e:	66 90                	xchg   %ax,%ax

0000000000002e90 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_>:
    2e90:	41 57                	push   %r15
    2e92:	41 56                	push   %r14
    2e94:	41 55                	push   %r13
    2e96:	41 54                	push   %r12
    2e98:	55                   	push   %rbp
    2e99:	53                   	push   %rbx
    2e9a:	48 83 ec 58          	sub    $0x58,%rsp
    2e9e:	48 85 ff             	test   %rdi,%rdi
    2ea1:	0f 84 5a 02 00 00    	je     3101 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x271>
    2ea7:	48 89 fb             	mov    %rdi,%rbx
    2eaa:	49 89 f4             	mov    %rsi,%r12
    2ead:	49 89 d5             	mov    %rdx,%r13
    2eb0:	0f 88 61 02 00 00    	js     3117 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x287>
    2eb6:	66 0f ef f6          	pxor   %xmm6,%xmm6
    2eba:	f2 48 0f 2a f7       	cvtsi2sd %rdi,%xmm6
    2ebf:	66 48 0f 7e f5       	movq   %xmm6,%rbp
    2ec4:	45 31 ff             	xor    %r15d,%r15d
    2ec7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2ece:	00 00 
    2ed0:	66 48 0f 6e c5       	movq   %rbp,%xmm0
    2ed5:	e8 66 f3 ff ff       	call   2240 <log2@plt>
    2eda:	f2 0f 2c f8          	cvttsd2si %xmm0,%edi
    2ede:	85 ff                	test   %edi,%edi
    2ee0:	0f 8e 2a 02 00 00    	jle    3110 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x280>
    2ee6:	4c 89 f9             	mov    %r15,%rcx
    2ee9:	31 d2                	xor    %edx,%edx
    2eeb:	31 c0                	xor    %eax,%eax
    2eed:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2ef4:	00 00 00 00 
    2ef8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2eff:	00 
    2f00:	48 89 ce             	mov    %rcx,%rsi
    2f03:	48 01 c0             	add    %rax,%rax
    2f06:	83 c2 01             	add    $0x1,%edx
    2f09:	48 d1 e9             	shr    $1,%rcx
    2f0c:	83 e6 01             	and    $0x1,%esi
    2f0f:	48 09 f0             	or     %rsi,%rax
    2f12:	39 d7                	cmp    %edx,%edi
    2f14:	75 ea                	jne    2f00 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x70>
    2f16:	48 c1 e0 03          	shl    $0x3,%rax
    2f1a:	49 8b 14 24          	mov    (%r12),%rdx
    2f1e:	4d 8b 45 00          	mov    0x0(%r13),%r8
    2f22:	f3 0f 10 04 02       	movss  (%rdx,%rax,1),%xmm0
    2f27:	f3 43 0f 11 04 f8    	movss  %xmm0,(%r8,%r15,8)
    2f2d:	f3 0f 10 44 02 04    	movss  0x4(%rdx,%rax,1),%xmm0
    2f33:	f3 43 0f 11 44 f8 04 	movss  %xmm0,0x4(%r8,%r15,8)
    2f3a:	49 83 c7 01          	add    $0x1,%r15
    2f3e:	4c 39 fb             	cmp    %r15,%rbx
    2f41:	75 8d                	jne    2ed0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x40>
    2f43:	49 83 ff 01          	cmp    $0x1,%r15
    2f47:	0f 84 b4 01 00 00    	je     3101 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x271>
    2f4d:	f3 0f 10 2d af 10 00 	movss  0x10af(%rip),%xmm5        # 4004 <_IO_stdin_used+0x4>
    2f54:	00 
    2f55:	f3 0f 10 25 ab 10 00 	movss  0x10ab(%rip),%xmm4        # 4008 <_IO_stdin_used+0x8>
    2f5c:	00 
    2f5d:	bf 02 00 00 00       	mov    $0x2,%edi
    2f62:	f3 0f 10 35 a2 10 00 	movss  0x10a2(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    2f69:	00 
    2f6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2f70:	41 89 fd             	mov    %edi,%r13d
    2f73:	41 d1 fd             	sar    $1,%r13d
    2f76:	83 ff 01             	cmp    $0x1,%edi
    2f79:	0f 8e 11 01 00 00    	jle    3090 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x200>
    2f7f:	48 63 d7             	movslq %edi,%rdx
    2f82:	49 63 cd             	movslq %r13d,%rcx
    2f85:	4c 89 c6             	mov    %r8,%rsi
    2f88:	48 8d 04 d5 00 00 00 	lea    0x0(,%rdx,8),%rax
    2f8f:	00 
    2f90:	49 8d 0c c8          	lea    (%r8,%rcx,8),%rcx
    2f94:	49 89 d6             	mov    %rdx,%r14
    2f97:	48 29 c6             	sub    %rax,%rsi
    2f9a:	48 29 c1             	sub    %rax,%rcx
    2f9d:	0f 1f 00             	nopl   (%rax)
    2fa0:	4a 8d 1c f5 00 00 00 	lea    0x0(,%r14,8),%rbx
    2fa7:	00 
    2fa8:	66 0f ef db          	pxor   %xmm3,%xmm3
    2fac:	0f 28 d6             	movaps %xmm6,%xmm2
    2faf:	45 31 e4             	xor    %r12d,%r12d
    2fb2:	48 8d 2c 33          	lea    (%rbx,%rsi,1),%rbp
    2fb6:	48 01 cb             	add    %rcx,%rbx
    2fb9:	eb 0b                	jmp    2fc6 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x136>
    2fbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2fc0:	0f 28 d8             	movaps %xmm0,%xmm3
    2fc3:	0f 28 d1             	movaps %xmm1,%xmm2
    2fc6:	f3 44 0f 10 13       	movss  (%rbx),%xmm10
    2fcb:	f3 0f 10 4b 04       	movss  0x4(%rbx),%xmm1
    2fd0:	f3 44 0f 10 45 00    	movss  0x0(%rbp),%xmm8
    2fd6:	f3 0f 10 7d 04       	movss  0x4(%rbp),%xmm7
    2fdb:	45 0f 28 ca          	movaps %xmm10,%xmm9
    2fdf:	0f 28 c1             	movaps %xmm1,%xmm0
    2fe2:	45 0f 28 da          	movaps %xmm10,%xmm11
    2fe6:	f3 0f 59 c3          	mulss  %xmm3,%xmm0
    2fea:	f3 44 0f 59 ca       	mulss  %xmm2,%xmm9
    2fef:	f3 44 0f 59 db       	mulss  %xmm3,%xmm11
    2ff4:	f3 44 0f 5c c8       	subss  %xmm0,%xmm9
    2ff9:	0f 28 c1             	movaps %xmm1,%xmm0
    2ffc:	f3 0f 59 c2          	mulss  %xmm2,%xmm0
    3000:	f3 41 0f 58 c3       	addss  %xmm11,%xmm0
    3005:	41 0f 2e c1          	ucomiss %xmm9,%xmm0
    3009:	0f 8a 9c 01 00 00    	jp     31ab <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x31b>
    300f:	41 0f 28 c9          	movaps %xmm9,%xmm1
    3013:	f3 41 0f 58 c8       	addss  %xmm8,%xmm1
    3018:	f3 45 0f 5c c1       	subss  %xmm9,%xmm8
    301d:	f3 0f 11 4d 00       	movss  %xmm1,0x0(%rbp)
    3022:	0f 28 c8             	movaps %xmm0,%xmm1
    3025:	f3 0f 58 cf          	addss  %xmm7,%xmm1
    3029:	f3 0f 5c f8          	subss  %xmm0,%xmm7
    302d:	0f 28 c5             	movaps %xmm5,%xmm0
    3030:	f3 0f 59 c3          	mulss  %xmm3,%xmm0
    3034:	f3 0f 11 4d 04       	movss  %xmm1,0x4(%rbp)
    3039:	0f 28 cc             	movaps %xmm4,%xmm1
    303c:	f3 0f 59 ca          	mulss  %xmm2,%xmm1
    3040:	f3 0f 11 7b 04       	movss  %xmm7,0x4(%rbx)
    3045:	0f 28 fc             	movaps %xmm4,%xmm7
    3048:	f3 0f 59 fb          	mulss  %xmm3,%xmm7
    304c:	f3 44 0f 11 03       	movss  %xmm8,(%rbx)
    3051:	f3 0f 5c c8          	subss  %xmm0,%xmm1
    3055:	0f 28 c5             	movaps %xmm5,%xmm0
    3058:	f3 0f 59 c2          	mulss  %xmm2,%xmm0
    305c:	f3 0f 58 c7          	addss  %xmm7,%xmm0
    3060:	0f 2e c1             	ucomiss %xmm1,%xmm0
    3063:	0f 8a d4 00 00 00    	jp     313d <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x2ad>
    3069:	41 83 c4 01          	add    $0x1,%r12d
    306d:	48 83 c5 08          	add    $0x8,%rbp
    3071:	48 83 c3 08          	add    $0x8,%rbx
    3075:	45 39 e5             	cmp    %r12d,%r13d
    3078:	0f 8f 42 ff ff ff    	jg     2fc0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x130>
    307e:	4a 8d 04 32          	lea    (%rdx,%r14,1),%rax
    3082:	4d 39 fe             	cmp    %r15,%r14
    3085:	73 09                	jae    3090 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x200>
    3087:	49 89 c6             	mov    %rax,%r14
    308a:	e9 11 ff ff ff       	jmp    2fa0 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x110>
    308f:	90                   	nop
    3090:	01 ff                	add    %edi,%edi
    3092:	4c 89 04 24          	mov    %r8,(%rsp)
    3096:	48 63 c7             	movslq %edi,%rax
    3099:	49 39 c7             	cmp    %rax,%r15
    309c:	72 63                	jb     3101 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x271>
    309e:	66 0f ef c0          	pxor   %xmm0,%xmm0
    30a2:	f2 0f 10 0d 36 10 00 	movsd  0x1036(%rip),%xmm1        # 40e0 <_IO_stdin_used+0xe0>
    30a9:	00 
    30aa:	89 7c 24 10          	mov    %edi,0x10(%rsp)
    30ae:	f2 0f 2a c7          	cvtsi2sd %edi,%xmm0
    30b2:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
    30b6:	f2 0f 5a c9          	cvtsd2ss %xmm1,%xmm1
    30ba:	0f 28 c1             	movaps %xmm1,%xmm0
    30bd:	f3 0f 11 4c 24 30    	movss  %xmm1,0x30(%rsp)
    30c3:	e8 38 f0 ff ff       	call   2100 <cosf@plt>
    30c8:	f3 0f 10 4c 24 30    	movss  0x30(%rsp),%xmm1
    30ce:	0f 57 0d 1b 10 00 00 	xorps  0x101b(%rip),%xmm1        # 40f0 <_IO_stdin_used+0xf0>
    30d5:	f3 0f 11 44 24 0c    	movss  %xmm0,0xc(%rsp)
    30db:	0f 28 c1             	movaps %xmm1,%xmm0
    30de:	e8 3d f0 ff ff       	call   2120 <sinf@plt>
    30e3:	f3 0f 10 64 24 0c    	movss  0xc(%rsp),%xmm4
    30e9:	8b 7c 24 10          	mov    0x10(%rsp),%edi
    30ed:	4c 8b 04 24          	mov    (%rsp),%r8
    30f1:	f3 0f 10 35 13 0f 00 	movss  0xf13(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    30f8:	00 
    30f9:	0f 28 e8             	movaps %xmm0,%xmm5
    30fc:	e9 6f fe ff ff       	jmp    2f70 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0xe0>
    3101:	48 83 c4 58          	add    $0x58,%rsp
    3105:	5b                   	pop    %rbx
    3106:	5d                   	pop    %rbp
    3107:	41 5c                	pop    %r12
    3109:	41 5d                	pop    %r13
    310b:	41 5e                	pop    %r14
    310d:	41 5f                	pop    %r15
    310f:	c3                   	ret
    3110:	31 c0                	xor    %eax,%eax
    3112:	e9 03 fe ff ff       	jmp    2f1a <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x8a>
    3117:	48 89 f8             	mov    %rdi,%rax
    311a:	48 89 fa             	mov    %rdi,%rdx
    311d:	66 0f ef c0          	pxor   %xmm0,%xmm0
    3121:	48 d1 e8             	shr    $1,%rax
    3124:	83 e2 01             	and    $0x1,%edx
    3127:	48 09 d0             	or     %rdx,%rax
    312a:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
    312f:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
    3133:	66 48 0f 7e c5       	movq   %xmm0,%rbp
    3138:	e9 87 fd ff ff       	jmp    2ec4 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x34>
    313d:	0f 28 c4             	movaps %xmm4,%xmm0
    3140:	0f 28 cd             	movaps %xmm5,%xmm1
    3143:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
    3148:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
    314d:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
    3152:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
    3157:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    315b:	f3 0f 11 6c 24 0c    	movss  %xmm5,0xc(%rsp)
    3161:	f3 0f 11 24 24       	movss  %xmm4,(%rsp)
    3166:	e8 05 f1 ff ff       	call   2270 <__mulsc3@plt>
    316b:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    3170:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    3175:	66 0f 6f f0          	movdqa %xmm0,%xmm6
    3179:	66 0f 6f c8          	movdqa %xmm0,%xmm1
    317d:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
    3182:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    3187:	0f c6 f6 55          	shufps $0x55,%xmm6,%xmm6
    318b:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    318f:	66 0f 6f c6          	movdqa %xmm6,%xmm0
    3193:	f3 0f 10 6c 24 0c    	movss  0xc(%rsp),%xmm5
    3199:	f3 0f 10 35 6b 0e 00 	movss  0xe6b(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    31a0:	00 
    31a1:	f3 0f 10 24 24       	movss  (%rsp),%xmm4
    31a6:	e9 be fe ff ff       	jmp    3069 <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x1d9>
    31ab:	41 0f 28 c2          	movaps %xmm10,%xmm0
    31af:	48 89 54 24 48       	mov    %rdx,0x48(%rsp)
    31b4:	48 89 74 24 40       	mov    %rsi,0x40(%rsp)
    31b9:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
    31be:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
    31c3:	89 7c 24 30          	mov    %edi,0x30(%rsp)
    31c7:	f3 0f 11 6c 24 34    	movss  %xmm5,0x34(%rsp)
    31cd:	f3 0f 11 64 24 28    	movss  %xmm4,0x28(%rsp)
    31d3:	f3 44 0f 11 44 24 20 	movss  %xmm8,0x20(%rsp)
    31da:	f3 0f 11 7c 24 18    	movss  %xmm7,0x18(%rsp)
    31e0:	f3 0f 11 5c 24 0c    	movss  %xmm3,0xc(%rsp)
    31e6:	f3 0f 11 14 24       	movss  %xmm2,(%rsp)
    31eb:	e8 80 f0 ff ff       	call   2270 <__mulsc3@plt>
    31f0:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
    31f5:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
    31fa:	66 0f 6f f0          	movdqa %xmm0,%xmm6
    31fe:	66 44 0f 6f c8       	movdqa %xmm0,%xmm9
    3203:	8b 7c 24 30          	mov    0x30(%rsp),%edi
    3207:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    320c:	0f c6 f6 55          	shufps $0x55,%xmm6,%xmm6
    3210:	f3 0f 10 6c 24 34    	movss  0x34(%rsp),%xmm5
    3216:	66 0f 6f c6          	movdqa %xmm6,%xmm0
    321a:	f3 0f 10 64 24 28    	movss  0x28(%rsp),%xmm4
    3220:	f3 0f 10 35 e4 0d 00 	movss  0xde4(%rip),%xmm6        # 400c <_IO_stdin_used+0xc>
    3227:	00 
    3228:	f3 44 0f 10 44 24 20 	movss  0x20(%rsp),%xmm8
    322f:	f3 0f 10 7c 24 18    	movss  0x18(%rsp),%xmm7
    3235:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    323a:	f3 0f 10 5c 24 0c    	movss  0xc(%rsp),%xmm3
    3240:	f3 0f 10 14 24       	movss  (%rsp),%xmm2
    3245:	e9 c5 fd ff ff       	jmp    300f <_Z16fft_cooley_tukeymRSt6vectorISt7complexIfESaIS1_EES4_+0x17f>
    324a:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3251:	00 00 00 
    3254:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    325b:	00 00 00 
    325e:	66 90                	xchg   %ax,%ax

0000000000003260 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>:
    3260:	41 57                	push   %r15
    3262:	41 56                	push   %r14
    3264:	41 55                	push   %r13
    3266:	41 54                	push   %r12
    3268:	55                   	push   %rbp
    3269:	48 89 f5             	mov    %rsi,%rbp
    326c:	53                   	push   %rbx
    326d:	48 89 fb             	mov    %rdi,%rbx
    3270:	48 81 ec 48 02 00 00 	sub    $0x248,%rsp
    3277:	4c 8d ac 24 30 01 00 	lea    0x130(%rsp),%r13
    327e:	00 
    327f:	4c 8d 64 24 30       	lea    0x30(%rsp),%r12
    3284:	4c 89 ef             	mov    %r13,%rdi
    3287:	4c 89 6c 24 18       	mov    %r13,0x18(%rsp)
    328c:	e8 ef ed ff ff       	call   2080 <_ZNSt8ios_baseC2Ev@plt>
    3291:	4c 8b 35 f8 2a 00 00 	mov    0x2af8(%rip),%r14        # 5d90 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x8>
    3298:	31 d2                	xor    %edx,%edx
    329a:	31 f6                	xor    %esi,%esi
    329c:	48 8d 05 15 29 00 00 	lea    0x2915(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    32a3:	66 0f ef c0          	pxor   %xmm0,%xmm0
    32a7:	66 89 94 24 10 02 00 	mov    %dx,0x210(%rsp)
    32ae:	00 
    32af:	48 8b 0d e2 2a 00 00 	mov    0x2ae2(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    32b6:	0f 11 84 24 18 02 00 	movups %xmm0,0x218(%rsp)
    32bd:	00 
    32be:	0f 11 84 24 28 02 00 	movups %xmm0,0x228(%rsp)
    32c5:	00 
    32c6:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
    32cd:	00 
    32ce:	49 8b 46 e8          	mov    -0x18(%r14),%rax
    32d2:	48 c7 84 24 08 02 00 	movq   $0x0,0x208(%rsp)
    32d9:	00 00 00 00 00 
    32de:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    32e3:	48 89 4c 04 30       	mov    %rcx,0x30(%rsp,%rax,1)
    32e8:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    32ef:	00 00 
    32f1:	49 8b 7e e8          	mov    -0x18(%r14),%rdi
    32f5:	4c 01 e7             	add    %r12,%rdi
    32f8:	e8 b3 ee ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    32fd:	48 8d 05 7c 29 00 00 	lea    0x297c(%rip),%rax        # 5c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    3304:	4c 8d 7c 24 40       	lea    0x40(%rsp),%r15
    3309:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    330e:	4c 89 ff             	mov    %r15,%rdi
    3311:	48 83 c0 28          	add    $0x28,%rax
    3315:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
    331c:	00 
    331d:	e8 3e ee ff ff       	call   2160 <_ZNSt13basic_filebufIcSt11char_traitsIcEEC1Ev@plt>
    3322:	4c 89 fe             	mov    %r15,%rsi
    3325:	4c 89 ef             	mov    %r13,%rdi
    3328:	e8 83 ee ff ff       	call   21b0 <_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E@plt>
    332d:	48 8b 75 00          	mov    0x0(%rbp),%rsi
    3331:	ba 08 00 00 00       	mov    $0x8,%edx
    3336:	4c 89 ff             	mov    %r15,%rdi
    3339:	e8 f2 ed ff ff       	call   2130 <_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode@plt>
    333e:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
    3343:	48 8b 7a e8          	mov    -0x18(%rdx),%rdi
    3347:	4c 01 e7             	add    %r12,%rdi
    334a:	48 85 c0             	test   %rax,%rax
    334d:	0f 84 2d 02 00 00    	je     3580 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x320>
    3353:	31 f6                	xor    %esi,%esi
    3355:	e8 c6 ee ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    335a:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
    3361:	00 
    3362:	66 0f ef c0          	pxor   %xmm0,%xmm0
    3366:	4c 8d 6c 24 28       	lea    0x28(%rsp),%r13
    336b:	0f 11 03             	movups %xmm0,(%rbx)
    336e:	4c 89 ee             	mov    %r13,%rsi
    3371:	4c 89 e7             	mov    %r12,%rdi
    3374:	e8 17 ef ff ff       	call   2290 <_ZNSi10_M_extractIfEERSiRT_@plt>
    3379:	48 89 c7             	mov    %rax,%rdi
    337c:	48 8d 74 24 2c       	lea    0x2c(%rsp),%rsi
    3381:	e8 0a ef ff ff       	call   2290 <_ZNSi10_M_extractIfEERSiRT_@plt>
    3386:	48 8b 10             	mov    (%rax),%rdx
    3389:	48 8b 52 e8          	mov    -0x18(%rdx),%rdx
    338d:	f6 44 10 20 05       	testb  $0x5,0x20(%rax,%rdx,1)
    3392:	0f 85 48 01 00 00    	jne    34e0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x280>
    3398:	48 8b 53 08          	mov    0x8(%rbx),%rdx
    339c:	48 3b 53 10          	cmp    0x10(%rbx),%rdx
    33a0:	74 2e                	je     33d0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x170>
    33a2:	f3 0f 10 4c 24 2c    	movss  0x2c(%rsp),%xmm1
    33a8:	48 83 c2 08          	add    $0x8,%rdx
    33ac:	4c 89 ee             	mov    %r13,%rsi
    33af:	4c 89 e7             	mov    %r12,%rdi
    33b2:	f3 0f 10 44 24 28    	movss  0x28(%rsp),%xmm0
    33b8:	0f 14 c1             	unpcklps %xmm1,%xmm0
    33bb:	0f 13 42 f8          	movlps %xmm0,-0x8(%rdx)
    33bf:	48 89 53 08          	mov    %rdx,0x8(%rbx)
    33c3:	e8 c8 ee ff ff       	call   2290 <_ZNSi10_M_extractIfEERSiRT_@plt>
    33c8:	eb af                	jmp    3379 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x119>
    33ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    33d0:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    33d7:	ff ff 0f 
    33da:	48 8b 0b             	mov    (%rbx),%rcx
    33dd:	48 89 d6             	mov    %rdx,%rsi
    33e0:	48 29 ce             	sub    %rcx,%rsi
    33e3:	48 89 f5             	mov    %rsi,%rbp
    33e6:	48 c1 fd 03          	sar    $0x3,%rbp
    33ea:	48 39 c5             	cmp    %rax,%rbp
    33ed:	0f 84 12 02 00 00    	je     3605 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3a5>
    33f3:	48 85 ed             	test   %rbp,%rbp
    33f6:	b8 01 00 00 00       	mov    $0x1,%eax
    33fb:	48 0f 45 c5          	cmovne %rbp,%rax
    33ff:	48 01 c5             	add    %rax,%rbp
    3402:	0f 82 88 01 00 00    	jb     3590 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x330>
    3408:	48 b8 ff ff ff ff ff 	movabs $0xfffffffffffffff,%rax
    340f:	ff ff 0f 
    3412:	48 39 c5             	cmp    %rax,%rbp
    3415:	48 0f 47 e8          	cmova  %rax,%rbp
    3419:	48 c1 e5 03          	shl    $0x3,%rbp
    341d:	48 89 ef             	mov    %rbp,%rdi
    3420:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
    3425:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    342a:	48 89 14 24          	mov    %rdx,(%rsp)
    342e:	e8 0d ed ff ff       	call   2140 <_Znwm@plt>
    3433:	f3 0f 10 4c 24 2c    	movss  0x2c(%rsp),%xmm1
    3439:	f3 0f 10 44 24 28    	movss  0x28(%rsp),%xmm0
    343f:	49 89 c0             	mov    %rax,%r8
    3442:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
    3447:	48 8b 14 24          	mov    (%rsp),%rdx
    344b:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    3450:	0f 14 c1             	unpcklps %xmm1,%xmm0
    3453:	0f 13 04 30          	movlps %xmm0,(%rax,%rsi,1)
    3457:	48 39 ca             	cmp    %rcx,%rdx
    345a:	74 44                	je     34a0 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x240>
    345c:	48 29 ca             	sub    %rcx,%rdx
    345f:	48 89 ce             	mov    %rcx,%rsi
    3462:	48 01 c2             	add    %rax,%rdx
    3465:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    346c:	00 00 00 00 
    3470:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    3477:	00 00 00 00 
    347b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3480:	f3 0f 10 06          	movss  (%rsi),%xmm0
    3484:	48 83 c0 08          	add    $0x8,%rax
    3488:	48 83 c6 08          	add    $0x8,%rsi
    348c:	f3 0f 11 40 f8       	movss  %xmm0,-0x8(%rax)
    3491:	f3 0f 10 46 fc       	movss  -0x4(%rsi),%xmm0
    3496:	f3 0f 11 40 fc       	movss  %xmm0,-0x4(%rax)
    349b:	48 39 d0             	cmp    %rdx,%rax
    349e:	75 e0                	jne    3480 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x220>
    34a0:	48 83 c0 08          	add    $0x8,%rax
    34a4:	48 85 c9             	test   %rcx,%rcx
    34a7:	74 21                	je     34ca <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x26a>
    34a9:	48 8b 73 10          	mov    0x10(%rbx),%rsi
    34ad:	48 89 cf             	mov    %rcx,%rdi
    34b0:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
    34b5:	48 89 04 24          	mov    %rax,(%rsp)
    34b9:	48 29 ce             	sub    %rcx,%rsi
    34bc:	e8 8f ec ff ff       	call   2150 <_ZdlPvm@plt>
    34c1:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
    34c6:	48 8b 04 24          	mov    (%rsp),%rax
    34ca:	4c 89 03             	mov    %r8,(%rbx)
    34cd:	49 01 e8             	add    %rbp,%r8
    34d0:	48 89 43 08          	mov    %rax,0x8(%rbx)
    34d4:	4c 89 43 10          	mov    %r8,0x10(%rbx)
    34d8:	e9 91 fe ff ff       	jmp    336e <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x10e>
    34dd:	0f 1f 00             	nopl   (%rax)
    34e0:	48 8d 05 99 27 00 00 	lea    0x2799(%rip),%rax        # 5c80 <_ZTVSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x18>
    34e7:	4c 89 ff             	mov    %r15,%rdi
    34ea:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    34ef:	48 83 c0 28          	add    $0x28,%rax
    34f3:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
    34fa:	00 
    34fb:	48 8d 05 c6 27 00 00 	lea    0x27c6(%rip),%rax        # 5cc8 <_ZTVSt13basic_filebufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3502:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    3507:	e8 44 eb ff ff       	call   2050 <_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv@plt>
    350c:	48 8d bc 24 a8 00 00 	lea    0xa8(%rsp),%rdi
    3513:	00 
    3514:	e8 47 ed ff ff       	call   2260 <_ZNSt12__basic_fileIcED1Ev@plt>
    3519:	48 8d 05 b8 26 00 00 	lea    0x26b8(%rip),%rax        # 5bd8 <_ZTVSt15basic_streambufIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3520:	48 8d 7c 24 78       	lea    0x78(%rsp),%rdi
    3525:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    352a:	e8 91 ec ff ff       	call   21c0 <_ZNSt6localeD1Ev@plt>
    352f:	49 8b 46 e8          	mov    -0x18(%r14),%rax
    3533:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    3538:	48 8b 0d 59 28 00 00 	mov    0x2859(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    353f:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    3544:	48 89 4c 04 30       	mov    %rcx,0x30(%rsp,%rax,1)
    3549:	48 8d 05 68 26 00 00 	lea    0x2668(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    3550:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
    3557:	00 
    3558:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    355f:	00 00 
    3561:	e8 2a eb ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    3566:	48 81 c4 48 02 00 00 	add    $0x248,%rsp
    356d:	48 89 d8             	mov    %rbx,%rax
    3570:	5b                   	pop    %rbx
    3571:	5d                   	pop    %rbp
    3572:	41 5c                	pop    %r12
    3574:	41 5d                	pop    %r13
    3576:	41 5e                	pop    %r14
    3578:	41 5f                	pop    %r15
    357a:	c3                   	ret
    357b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3580:	8b 77 20             	mov    0x20(%rdi),%esi
    3583:	83 ce 04             	or     $0x4,%esi
    3586:	e8 95 ec ff ff       	call   2220 <_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate@plt>
    358b:	e9 ca fd ff ff       	jmp    335a <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0xfa>
    3590:	48 bd f8 ff ff ff ff 	movabs $0x7ffffffffffffff8,%rbp
    3597:	ff ff 7f 
    359a:	e9 7e fe ff ff       	jmp    341d <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1bd>
    359f:	48 89 c3             	mov    %rax,%rbx
    35a2:	eb 0d                	jmp    35b1 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x351>
    35a4:	48 89 c3             	mov    %rax,%rbx
    35a7:	eb 24                	jmp    35cd <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x36d>
    35a9:	4c 89 ff             	mov    %r15,%rdi
    35ac:	e8 2f ec ff ff       	call   21e0 <_ZNSt13basic_filebufIcSt11char_traitsIcEED1Ev@plt>
    35b1:	49 8b 46 e8          	mov    -0x18(%r14),%rax
    35b5:	48 8b 0d dc 27 00 00 	mov    0x27dc(%rip),%rcx        # 5d98 <_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    35bc:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    35c1:	48 89 4c 04 30       	mov    %rcx,0x30(%rsp,%rax,1)
    35c6:	31 c0                	xor    %eax,%eax
    35c8:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    35cd:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    35d2:	48 8d 05 df 25 00 00 	lea    0x25df(%rip),%rax        # 5bb8 <_ZTVSt9basic_iosIcSt11char_traitsIcEE@GLIBCXX_3.4+0x10>
    35d9:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
    35e0:	00 
    35e1:	e8 aa ea ff ff       	call   2090 <_ZNSt8ios_baseD2Ev@plt>
    35e6:	48 89 df             	mov    %rbx,%rdi
    35e9:	e8 42 ec ff ff       	call   2230 <_Unwind_Resume@plt>
    35ee:	48 89 c3             	mov    %rax,%rbx
    35f1:	eb b6                	jmp    35a9 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x349>
    35f3:	48 89 c7             	mov    %rax,%rdi
    35f6:	e8 a5 ea ff ff       	call   20a0 <__cxa_begin_catch@plt>
    35fb:	e8 10 ec ff ff       	call   2210 <__cxa_end_catch@plt>
    3600:	e9 07 ff ff ff       	jmp    350c <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2ac>
    3605:	48 8d 3d 08 0a 00 00 	lea    0xa08(%rip),%rdi        # 4014 <_IO_stdin_used+0x14>
    360c:	e8 af ea ff ff       	call   20c0 <_ZSt20__throw_length_errorPKc@plt>
    3611:	48 89 c5             	mov    %rax,%rbp
    3614:	48 8b 3b             	mov    (%rbx),%rdi
    3617:	48 8b 73 10          	mov    0x10(%rbx),%rsi
    361b:	48 29 fe             	sub    %rdi,%rsi
    361e:	48 85 ff             	test   %rdi,%rdi
    3621:	74 05                	je     3628 <_Z17read_complex_dataRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3c8>
    3623:	e8 28 eb ff ff       	call   2150 <_ZdlPvm@plt>
    3628:	4c 89 e7             	mov    %r12,%rdi
    362b:	e8 40 ea ff ff       	call   2070 <_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev@plt>
    3630:	48 89 ef             	mov    %rbp,%rdi
    3633:	e8 f8 eb ff ff       	call   2230 <_Unwind_Resume@plt>

Disassembly of section .fini:

0000000000003638 <_fini>:
    3638:	48 83 ec 08          	sub    $0x8,%rsp
    363c:	48 83 c4 08          	add    $0x8,%rsp
    3640:	c3                   	ret
