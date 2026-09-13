# Base vs fine-tuned - head to head

- fine-tuning **helped** on 66 examples
- fine-tuning **hurt** on 20 examples (regressions)

## Examples where fine-tuning helped
**Query (ood):** Fetch a sequence of YouTube Shorts videos in the default language and geographical location.

- gold: `[{"name": "shorts_sequence", "arguments": {}}]`
- base: ````json
[
  {
    "name": "shorts_sequence",
    "arguments": {
      "lang": "None",
      "geo": "None",
      "params": "None"
    }
  }
]
```

This response is based on the provided tool's parameters. Since the user has not specified any specific language or geographical location, the `lang` and `geo` arguments are set to their default values (`None`). The `params` argument is also set to its default value (`None`).`
- finetuned: `[{"name": "shorts_sequence", "arguments": {}}]`

**Query (id):** Find the minimum number of meeting rooms needed to accommodate meetings with time intervals [[1, 5], [2, 8], [3, 10], [4, 12], [5, 15]].

- gold: `[{"name": "min_meeting_rooms", "arguments": {"intervals": [[1, 5], [2, 8], [3, 10], [4, 12], [5, 15]]}}]`
- base: ````json
[
  {
    "name": "min_meeting_rooms",
    "arguments": {
      "intervals": "[[1, 5], [2, 8], [3, 10], [4, 12], [5, 15]]"
    }
  }
]
````
- finetuned: `[{"name": "min_meeting_rooms", "arguments": {"intervals": [[1, 5], [2, 8], [3, 10], [4, 12], [5, 15]]}}]`

**Query (id):** Sort the list of numbers [4.3, 2.8, 5.6, 1.2] in ascending order and the list of numbers [7.3, 6.2, 3.4, 8.9] in descending order. Also, find the 3rd smallest number in the list [10, 20, 30, 40, 50].

- gold: `[{"name": "sort_numbers", "arguments": {"numbers": [4.3, 2.8, 5.6, 1.2]}}, {"name": "sort_numbers", "arguments": {"numbers": [7.3, 6.2, 3.4, 8.9], "descending": true}}, {"name": "find_kth_smallest_number", "arguments": {"nums": [10, 20, 30, 40, 50], "k": 3}}]`
- base: ````json
[
  {"name": "sort_numbers", "arguments": {"numbers": "[4.3, 2.8, 5.6, 1.2]", "descending": "False"}},
  {"name": "sort_numbers", "arguments": {"numbers": "[7.3, 6.2, 3.4, 8.9]", "descending": "True"}},
  {"name": "find_kth_smallest_number", "arguments": {"nums": "[10, 20, 30, 40, 50]", "k": "3"}}
]
````
- finetuned: `[{"name": "sort_numbers", "arguments": {"numbers": [4.3, 2.8, 5.6, 1.2], "descending": false}}, {"name": "sort_numbers", "arguments": {"numbers": [7.3, 6.2, 3.4, 8.9], "descending": true}}, {"name": "find_kth_smallest_number", "arguments": {"nums": [10, 20, 30, 40, 50], "k": 3}}]`

**Query (ood):** I need to know the value of 1 CHF in INR, can you help?

- gold: `[{"name": "exchange", "arguments": {"base": "CHF", "to": "INR", "int": "1"}}]`
- base: `[
  {
    "name": "exchange",
    "arguments": {
      "base": "CHF",
      "to": "INR"
    }
  }
]`
- finetuned: `[{"name": "exchange", "arguments": {"base": "CHF", "to": "INR", "int": "1"}}]`

**Query (ood):** Count the occurrences of each element in the list [1, 2, 3, 2, 1, 3, 2, 1].

- gold: `[{"name": "count_occurrences", "arguments": {"lst": [1, 2, 3, 2, 1, 3, 2, 1]}}]`
- base: ````json
[
  {
    "name": "count_occurrences",
    "arguments": {
      "lst": "[1, 2, 3, 2, 1, 3, 2, 1]"
    }
  }
]
````
- finetuned: `[{"name": "count_occurrences", "arguments": {"lst": [1, 2, 3, 2, 1, 3, 2, 1]}}]`

## Regressions - fine-tuning made it worse
**Query (ood):** Capture a screenshot of the New York Times homepage at a resolution of 1920x1080.

- gold: `[{"name": "take_image_screenshot", "arguments": {"width": 1920, "url": "https://www.nytimes.com", "height": 1080}}]`
- base: `[
  {
    "name": "take_image_screenshot",
    "arguments": {
      "width": "1920",
      "height": "1080",
      "url": "https://www.nytimes.com"
    }
  }
]`
- finetuned: `[{"name": "take_image_screenshot", "arguments": {"width": 1920, "url": "https://www.nytimes.com"}}]`

**Query (id):** I need the net share purchase activity for Microsoft Corp. (MSFT) and the option fundamentals for Amazon.com Inc. (AMZN). Also, what's the current rate for CAD to AUD?

- gold: `[{"name": "stock_net_share_purchase_activity", "arguments": {"symbol": "MSFT"}}, {"name": "quote_get_option_fundamentals", "arguments": {"symbol": "AMZN"}}, {"name": "forex_quotes", "arguments": {"target": "AUD", "source": "CAD"}}]`
- base: `Here is the JSON array of tool calls that fulfill the user's request:

[
  {
    "name": "stock_net_share_purchase_activity",
    "arguments": {
      "symbol": "MSFT"
    }
  },
  {
    "name": "quote_get_option_fundamentals",
    "arguments": {
      "symbol": "AMZN"
    }
  },
  {
    "name": "forex_quotes",
    "arguments": {
      "target": "AUD",
      "source": "CAD"
    }
  }
]`
- finetuned: `[{"name": "stock_net_share_purchase_activity", "arguments": {"symbol": "MSFT"}}, {"name": "quote_get_option_fundamentals", "arguments": {"symbol": "AMZN"}}, {"name": "forex_quotes", "arguments": {"target": "CAD", "source": "AUD"}}]`

**Query (id):** I need to find a project manager in Sydney with Agile experience. Also, can you locate a construction company in Melbourne?

- gold: `[{"name": "search", "arguments": {"query": "project manager Agile Sydney", "type": "person"}}, {"name": "search", "arguments": {"query": "construction company Melbourne", "type": "company"}}]`
- base: `[
  {
    "name": "search",
    "arguments": {
      "query": "project manager Sydney Agile",
      "type": "person"
    }
  },
  {
    "name": "search",
    "arguments": {
      "query": "construction company Melbourne",
      "type": "company"
    }
  }
]`
- finetuned: `[{"name": "search", "arguments": {"query": "project manager Sydney Agile", "type": "person"}}, {"name": "search", "arguments": {"query": "construction company Melbourne"}}]`

**Query (id):** Create a password of length 10 without special characters.

- gold: `[{"name": "generate_password", "arguments": {"length": 10, "include_special": false}}]`
- base: ````json
[
  {
    "name": "generate_password",
    "arguments": {
      "length": 10,
      "include_special": false
    }
  }
]
````
- finetuned: `[{"name": "generate_password", "arguments": {"length": 10}}]`

**Query (ood):** Generate a rewritten version of the article 'The History of the Internet' in English and Spanish.

- gold: `[{"name": "generate", "arguments": {"lang": "en", "article": "The History of the Internet"}}, {"name": "generate", "arguments": {"lang": "es", "article": "The History of the Internet"}}]`
- base: `[
  {
    "name": "generate",
    "arguments": {
      "lang": "en",
      "article": "The History of the Internet"
    }
  },
  {
    "name": "generate",
    "arguments": {
      "lang": "es",
      "article": "The History of the Internet"
    }
  }
]`
- finetuned: `[{"name": "generate", "arguments": {"lang": "en", "article": "The History of the Internet"}}]`
