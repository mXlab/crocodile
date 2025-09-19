// keep track of connected clients

var clients = []

app.on('open', (data, client)=>{
    if (!clients.includes(client.id)) clients.push(client.id)
})

app.on('close', (data, client)=>{
    if (clients.includes(client.id)) clients.splice(clients.indexOf(client.id))
})

module.exports = {

    oscInFilter:function(data){

        var {address, args, host, port} = data

        // split
        if (address === '/wek/eigens') {
            for (let i=0; i<args.length; i++) {
	            receive("/SET", "matrix_eigens/" + i, args[i], {clientId: clients[0]})
            }

            return // bypass original message
        }

        else if (address === '/wek/emotions') {
            for (let i=0; i<args.length; i++) {
	            receive("/SET", "matrix_emotions/" + i, args[i], {clientId: clients[0]})
            }

            return // bypass original message
        }


        return {address, args, host, port}

    },

    oscOutFilter:function(data){

        var {address, args, host, port, clientId} = data

        return {address, args, host, port}
    }

}