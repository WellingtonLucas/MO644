#!/bin/bash

execucao=0

while true
do
	for t in 1 2 3; do
		#echo "/srv/mo644/T-Cuda$t"
		if [ ! -d /srv/mo644/T-Cuda$t ]
		then
			mkdir /srv/mo644/T-Cuda$t
			if [ "$?" -eq 0 ]
			then
				#echo "Preparando Execução -- /srv/mo644/T-Cuda$t!"
				#Editar apenas a linha abaixo com o executável.
				./histogram arq1.ppm  #VC SÓ MODIFICARÁ AQUI
				execucao=1		
				rmdir /srv/mo644/T-Cuda$t		
				break #usar no final para sair do for
			fi
		fi
	done
	
	if [ "$execucao" -ge 1 ]
	then
		#echo "Finalizando execucao!"
		break
	else
		sleep 1
	fi
done

#echo "...done."
