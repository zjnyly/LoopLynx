// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.

`ifndef REPLACE_WITH_MODULE_TRACE_BASE_COMPONENT__SV
`define REPLACE_WITH_MODULE_TRACE_BASE_COMPONENT__SV
class REPLACE_WITH_MODULE_data_base_object;
endclass

class REPLACE_WITH_MODULE_trace_base_component;

string inst_name;
string hier_name;
static string TVDir;

REPLACE_WITH_MODULE_trace_base_component puts[$];
REPLACE_WITH_MODULE_trace_base_component gets[$];
REPLACE_WITH_MODULE_trace_base_component hooks[$];

function new(string inst_name, string hier_name);
    this.inst_name = inst_name;
    this.hier_name = hier_name;
endfunction

function string full_name();
    return {hier_name, ".", inst_name};
endfunction

function void add_puts(REPLACE_WITH_MODULE_trace_base_component putc);
    puts.push_back(putc);
endfunction
function void remove_puts(REPLACE_WITH_MODULE_trace_base_component putc);
    for(int i=0; i<puts.size; ) begin
        if( puts[i] == putc ) begin
            puts.delete(i);
        end else i++;
    end
endfunction

function void add_gets(REPLACE_WITH_MODULE_trace_base_component getc);
    gets.push_back(getc);
endfunction
function void remove_gets(REPLACE_WITH_MODULE_trace_base_component getc);
    for(int i=0; i<gets.size; ) begin
        if( gets[i] == getc ) begin
            gets.delete(i);
        end else i++;
    end
endfunction

function void add_hooks(REPLACE_WITH_MODULE_trace_base_component hookc);
    hooks.push_back(hookc);
endfunction
function void remove_hooks(REPLACE_WITH_MODULE_trace_base_component hookc);
    for(int i=0; i<hooks.size; ) begin
        if( hooks[i] == hookc ) begin
            hooks.delete(i);
        end else i++;
    end
endfunction

virtual function void put_data(REPLACE_WITH_MODULE_data_base_object data, string flag);
    foreach(puts[i]) puts[i].put_data_called(data, flag);
endfunction
virtual function void put_data_called(REPLACE_WITH_MODULE_data_base_object data, string flag);
endfunction

virtual function void get_data(REPLACE_WITH_MODULE_data_base_object data, string flag);
    foreach(gets[i]) gets[i].get_data_called(data, flag);
endfunction
virtual function void get_data_called(REPLACE_WITH_MODULE_data_base_object data, string flag);
endfunction

virtual function void hook_data(REPLACE_WITH_MODULE_data_base_object data, string flag);
    foreach(hooks[i]) hooks[i].hook_data_called(data, flag);
endfunction
virtual function void hook_data_called(REPLACE_WITH_MODULE_data_base_object data, string flag);
endfunction

virtual task run();
endtask
endclass
`endif
