dim=(
	1000
	10000
	100000
	1000000
	10000000
	100000000
	1000000000
	10000000000
)

var_pairs=(
	"QUANT_ALGO=UNIFORM SEND_ALGO=REC_HALVING BITS_VAR=8"
	"QUANT_ALGO=UNIFORM SEND_ALGO=REC_HALVING BITS_VAR=16"
	"QUANT_ALGO=HOMOMORPHIC SEND_ALGO=REC_HALVING BITS_VAR=8"
	"QUANT_ALGO=HOMOMORPHIC SEND_ALGO=REC_HALVING BITS_VAR=16"
	"QUANT_ALGO=HOMOMORPHIC SEND_ALGO=RING BITS_VAR=8"
	"QUANT_ALGO=HOMOMORPHIC SEND_ALGO=RING BITS_VAR=16"
	"QUANT_ALGO=KNOWN_RANGE SEND_ALGO=REC_HALVING BITS_VAR=8"
	"QUANT_ALGO=KNOWN_RANGE SEND_ALGO=REC_HALVING BITS_VAR=16"
	"QUANT_ALGO=KNOWN_RANGE SEND_ALGO=RING BITS_VAR=8"
	"QUANT_ALGO=KNOWN_RANGE SEND_ALGO=RING BITS_VAR=16"
	"QUANT_ALGO=NON_LINEAR SEND_ALGO=REC_HALVING BITS_VAR=8 NON_LINEAR_TYPE=1"
	"QUANT_ALGO=NON_LINEAR SEND_ALGO=REC_HALVING BITS_VAR=8 NON_LINEAR_TYPE=2"
	"QUANT_ALGO=LLOYD SEND_ALGO=REC_HALVING BITS_VAR=8"
)

for env_vars in "${var_pairs[@]}"; do
	IFS=' ' read -ra env_array <<<"$env_vars"

	export "${env_array[@]}"
	echo "${env_array[@]}"

	for arg in "${dim[@]}"; do
		# Calculate the number of bytes of the run
		bytes=$((arg * 4))
		#Convert bytes to KB, MB, or GB if needed
		if [ "$bytes" -ge 1000000000 ]; then
			bytes_str="$((bytes / 1000000000)) GB"
		elif [ "$bytes" -ge 1000000 ]; then
			bytes_str="$((bytes / 1000000)) MB"
		elif [ "$bytes" -ge 1000 ]; then
			bytes_str="$((bytes / 1000)) KB"
		else
			bytes_str="$bytes bytes"
		fi

		echo "Running with dimension: $arg ($bytes_str)"
		echo "Environmental variables set to $env_vars"

		echo "Number of nodes: 2"
		mpiexec -n 2 ./out "$arg"
		echo "Number of nodes: 4"
		mpiexec -n 4 ./out "$arg"
		echo "Number of nodes: 8"
		mpiexec -n 8 ./out "$arg"
		echo "Number of nodes: 16"
		mpiexec -n 16 ./out "$arg"
	done
done
