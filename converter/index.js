// NEEDS OLD VERSION OF graphql-compose (5.10)
const { composeWithMysql } = require("graphql-compose-mysql")
const graphql = require('graphql');
const fs = require('fs');

async function main() {
    return composeWithMysql({
        mysqlConfig: {
            host: "localhost",
            port: 3306,
            user: "root",
            password: "graphql",
            database: "graphql"
        },
    }).then(schema => {
		const data = graphql.printSchema(schema) 
		console.log(data);
		fs.writeFileSync('schema.graphql', data)
    }).catch(e => {
		console.log(e)
	})
}

main()
