// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.

`ifndef FILE_AGENT_PKG__SV
    `define FILE_AGENT_PKG__SV

    package file_agent_pkg;
        
        import uvm_pkg::*;
        `include "uvm_macros.svh"
        `include "file_read_agent.sv"
        `include "file_write_agent.sv"
        `include "mem_model.sv"
    endpackage

`endif
