start_time="$(date -u +%s)"

python -m a_select_tiny_data
echo "done with first one"
python -m b_create_embedding
echo "done with second one"
python -m c_concatenate_vectors
echo "done with third one"
python -m d_create_kfold 
echo "done with fourth one" 
python -m lr_embedding_for_blend
echo "done lr"
python -m xgb_embedding_blend
echo "done xgb"
python -m GNB_embedding_blend
echo "done GNB"
python -m lr_embedding_bagging_for_blend
echo "done lr bagging"
python -m blend_with_lr

end_time="$(date -u +%s)"

elapsed="$(($end_time-$start_time))"

echo "Total of $elapsed seconds elapsed for process"
