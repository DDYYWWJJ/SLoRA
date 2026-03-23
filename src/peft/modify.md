在src/peft/peft_model.py中添加代码
* PeftModel的from_pretrained函数中,为模型增加结构信息
在src/peft/mapping.py中添加代码
* get_peft_model函数中,为模型增加结构信息
在src/peft/tuners/lora/layer.py中:
* 添加LoraMatrixManager类,索引各层A,B矩阵信息
* LoraLayer:
  * LoraLayer中添加offset,start_layer,matrix_type,layer_idx四个属性
  * 添加property的注解函数
  * _infer_module_type,_infer_layer_idx函数
  * 修改update_layer函数
* Linear类:
  * 添加_get_combined_matrices函数获取合并A,B权重
  * 修改get_delta_weight,获取$\delta$并权重
  * 修改forward函数,使用合并的A,B权重