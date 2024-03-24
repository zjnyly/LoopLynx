array set opt {
    reset      0
    csim       0
    synth      1
    cosim      0
    validation 0
    export     0
    vsynth     0
    fifo_opt   0
}


foreach arg $::argv {
    foreach o [lsort [array names opt]] {
        regexp "$o=+(\\w+)" $arg unused opt($o)
    }
}

proc report_time { op_name time_start time_end } {
    set time_taken [expr $time_end - $time_start]
    set time_s [expr ($time_taken / 1000) % 60]
    set time_m [expr ($time_taken / (1000*60)) % 60]
    set time_h [expr ($time_taken / (1000*60*60)) % 24]
    puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}



file mkdir tb_data
set CSIM_RESULTS "./tb_data/csim_results_hls.log"
set RTL_COSIM_RESULTS "./tb_data/rtl_cosim_results_hls.log"



set kernel_name "LoopLynx"


if {$opt(reset)} {
    open_project -reset ${kernel_name}_prj
} else {
    open_project ${kernel_name}_prj
}

set src_path_root [pwd]

set INCLUDE_FLAGS "-I$src_path_root/include"  

#! 设置要综合的函数
set_top ${kernel_name}
add_files ./attention.cpp -cflags $INCLUDE_FLAGS
add_files ./attention.cpp -cflags "-std=c++0x"
add_files -tb attention_tb.cpp



if {$opt(reset)} {
    open_solution -reset "solution1"
} else {
    open_solution "solution1"
}

catch {config_array_partition -maximum_size 4096}
config_compile -name_max_length 80
set_part {xcvu190-flga2577-2-e}
#config_schedule -enable_dsp_full_reg=false
create_clock -period 10 -name default
set_clock_uncertainty 12.5%  default




if {$opt(csim)} {
    puts "***** C SIMULATION *****"
    set time_start [clock clicks -milliseconds]
    csim_design
    set time_end [clock clicks -milliseconds]
    report_time "C SIMULATION" $time_start $time_end
}

if {$opt(synth)} {
    puts "***** C/RTL SYNTHESIS *****"
    set time_start [clock clicks -milliseconds]
    csynth_design
    set time_end [clock clicks -milliseconds]
    report_time "C/RTL SYNTHESIS" $time_start $time_end
}


if {$opt(export)} {
    puts "***** EXPORT IP *****"
    set time_start [clock clicks -milliseconds]
    export_design -format ip_catalog
    set time_end [clock clicks -milliseconds]
    report_time "EXPORT IP" $time_start $time_end
}


exit
