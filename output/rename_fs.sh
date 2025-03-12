for f in ./elbaff_experiment/final_states/gpt-4o-mini*.json; do
    dir=$(dirname "$f")
    base=$(basename "$f")
    newbase="${base//gpt-4o-mini/gpt-4o-mini_}"
    mv "$f" "$dir/$newbase"
done

