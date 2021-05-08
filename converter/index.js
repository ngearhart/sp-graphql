const { ApolloServer } = require("apollo-server")
const { composeWithMysql } = require("graphql-compose-mysql")
const graphql = require('graphql');

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
		console.log(schema)
		console.log(graphql.printSchema(schema));
		// var root = {
		// 	hello: () => {
		// 	  return 'Hello world!';
		// 	},
		//   };
		// var app = express();
		// app.use('/graphql', graphqlHTTP({
		// 	schema: schema,
		// 	rootValue: root,
		// 	graphiql: true,
		// }));
		// app.listen(4000);
		// console.log('Running a GraphQL API server at http://localhost:4000/graphql');
		// console.log(schema)
		// console.log(schema.query)
		// const RootQuery = new graphql.GraphQLObjectType({
		// 	name: 'RootQueryType',
		// 	fields: {
		// 		status: {
		// 			type: graphql.GraphQLString,
		// 			resolve(parent, args) {
		// 				return "Welcome to GraphQL"
		// 			}
		// 		}
		// 	}
		// })
        // const server = new ApolloServer({
        //     schema: schema,
		// 	rootValue: RootQuery,
        //     playground: true,
        // })

        // server.listen().then(({ url }) => {
        //     console.log(`🚀 Server ready at ${url}`)
        // })
    }).catch(e => {
		console.log(e)
	})
}

main()