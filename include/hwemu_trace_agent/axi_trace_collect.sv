// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.

`ifndef REPLACE_WITH_MODULE_AXI_TRACE_COLLECT__SV
`define REPLACE_WITH_MODULE_AXI_TRACE_COLLECT__SV
class REPLACE_WITH_MODULE_axi_trace_collect extends REPLACE_WITH_MODULE_trace_base_component;

    REPLACE_WITH_MODULE_axi_trace_data awq        [*][$];
    REPLACE_WITH_MODULE_axi_trace_data wq         [*][$];
    //REPLACE_WITH_MODULE_axi_trace_data bq_rsp     [*][$];
    //REPLACE_WITH_MODULE_axi_trace_data bq_a2rinfo [][$];

    REPLACE_WITH_MODULE_axi_trace_data arq[*][$];
    string mode;

    function new(string inst_name, string hier_name);
        super.new(inst_name, hier_name);
    endfunction

    virtual function void cfg(string mode);
        this.mode = mode;
    endfunction

    virtual function bit ret_brsp_chk(int id);
        REPLACE_WITH_MODULE_axi_trace_data brsp;
        if(awq[id].size == wq[id].size && wq[id][$].last) begin
            if(awq[id][0].len != wq[id][0].strb.size) begin
                $display("Error, last bit or data size/len unmatch err in %s at %t. TB file:%s line:%d", this.full_name(), $time, `__FILE__, `__LINE__ );
                $display(awq[id][0].sprint() );
                $display(wq[id][0].sprint() );
            end
            brsp = awq[id][0].clone(); 
            put_data(brsp, "aww_both_rx"); //put a null data as no data info is needed for brsp
        end
    endfunction

    virtual function void write_aw(REPLACE_WITH_MODULE_axi_trace_data tr);
//$display("lineww, awrx in %s at %t. TB file:%s line:%d", this.full_name(), $time, `__FILE__, `__LINE__ );
        awq[tr.id].push_back( tr.clone() );
        ret_brsp_chk(tr.id);
    endfunction


    virtual function void write_w(REPLACE_WITH_MODULE_axi_trace_data tr);
        REPLACE_WITH_MODULE_axi_trace_data pop_wq_tr;	
//$display("lineww, wrx in %s at %t. TB file:%s line:%d", this.full_name(), $time, `__FILE__, `__LINE__ );
        if(wq[tr.id].size==0|| wq[tr.id][$].last) 
            wq[tr.id].push_back( tr.clone() );
        else wq[tr.id][$].combine_dinfo(tr);
        ret_brsp_chk(tr.id);
    endfunction

    virtual function void write_b(REPLACE_WITH_MODULE_axi_trace_data tr);
        REPLACE_WITH_MODULE_axi_trace_data axi_tr;
        axi_tr = wq[tr.id].pop_front();
        //axi_tr = wq[tr.id][0];
        axi_tr.combine_ainfo( awq[tr.id].pop_front() );
        //axi_tr.combine_ainfo( awq[tr.id][0] );
        axi_tr.combine_binfo( tr );
        if(axi_tr.len != axi_tr.strb.size || axi_tr.last!=1) begin
            $display("Error, last bit or data size/len unmatch err in %s at %t. TB file:%s line:%d", this.full_name(), $time, `__FILE__, `__LINE__ );
            //$display(awq[tr.id][0].sprint() );
            //$display(wq[tr.id][0].sprint() );
            //$display(axi_tr.sprint() );
        end
        //wq[tr.id].pop_front(); awq[tr.id].pop_front() ;
        axi_tr.waddrdata_strobe_infer();
        put_data( axi_tr, "axi_brx_end");
    endfunction

    virtual function void write_ar(REPLACE_WITH_MODULE_axi_trace_data tr);
        REPLACE_WITH_MODULE_axi_trace_data read_rsp;
        arq[tr.id].push_back(tr.clone());
        read_rsp = tr.clone();
        get_data(read_rsp, "fetch_rdata" );
        put_data(read_rsp, "feedback_rdata");
    endfunction

    virtual function void write_r(REPLACE_WITH_MODULE_axi_trace_data tr);
        arq[tr.id][0].combine_dinfo(tr);
        //$display("lineww rdata: %s", arq[tr.id][0].sprint());
        if(tr.last) begin
            //$display("lineww rdata beforeinfer: %s", arq[tr.id][0].sprint());
            arq[tr.id][0].raddrdata_infer();
            put_data(arq[tr.id].pop_front(), "axi_rrx_end");
        end
    endfunction

    virtual function void put_data_called(REPLACE_WITH_MODULE_data_base_object data, string flag);
        REPLACE_WITH_MODULE_axi_trace_data d_cast;
        if( !$cast(d_cast, data) ) $display("Error, REPLACE_WITH_MODULE_data_base_object data can't be casted into REPLACE_WITH_MODULE_axi_trace_data in %s at %t. TB file:%s line:%d", this.full_name(), $time, `__FILE__, `__LINE__ );

//$display("lineww, collect_rx  %s at %t. ", this.full_name(), $time);
//$display( d_cast.sprint() );
        case(flag)
            "aw_put" : write_aw(d_cast);
             "w_put" : write_w(d_cast);
             "b_put" : write_b(d_cast);
            "ar_put" : write_ar(d_cast);
            "r_put"  : write_r(d_cast);
            "rdcmd"  : rdcmd_trans2phy(d_cast);
            "wrcmd"  : wdcmd_trans2phy(d_cast);
            default : $display("Error, undefined put flag in %s at %t. TB file:%s line:%d", this.full_name(), $time, `__FILE__, `__LINE__ );
        endcase
    endfunction

    virtual function void wdcmd_trans2phy(REPLACE_WITH_MODULE_axi_trace_data tr);
        if(this.mode!="tvread") return;
        put_data(tr, "aw_master");
        put_data(tr, "w_master");
        put_data(tr, "b_master");
    endfunction

    virtual function void rdcmd_trans2phy(REPLACE_WITH_MODULE_axi_trace_data tr);
        if(this.mode!="tvread") return;
        put_data(tr, "ar_master");
        put_data(tr, "r_master");
    endfunction
endclass
`endif
