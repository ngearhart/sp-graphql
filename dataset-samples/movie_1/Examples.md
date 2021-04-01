## Example 1
Prompt: "Who was the director of the movie released most recently?" <br>
Query:
```
query {
  movie(limit: 1, order_by: {year: desc}) {
    director
  }
}
```

## Example 2
Prompt: "Which movies has Sarah Martinez reviewed?" <br>
Query:
```
query {
  movie(find: {rating_reviewer_name: "Sarah Martinez"}) {
    title
    rating {
      reviewer {
        name
      }
    }
  }
}
```
