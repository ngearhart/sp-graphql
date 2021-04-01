Note: This dataset calls a student a "list" for some reason<br>
## Example 1
Prompt: "What grade is Noah Gearhart in?" <br>
Query:
```
query {
  list(limit: 1, find: {firstname: "Noah", lastname: "Gearhart"}) {
    grade
    firstname
    lastname
  }
}
```

## Example 2
Prompt: "What students have Brad Kszastowski as their teacher?" <br>
Query:
```
query {
  list(find: {teacher_firstname: "Brad", teacher_lastname: "Kszastowski"}) {
    firstname
    lastname
    teacher {
      firstname
      lastname
    }
  }
}
```
