# Test Queries for AI Task Copilot

## GitHub-Only Queries (Existing Functionality)

### Working Test Cases
1. "What is issue number 3275" ✅
2. "Show me all open issues in the repository" ✅
3. "Find all issues related to 'performance' and show me the first 5" ✅
4. "Show me all open pull requests in the repository" ✅
5. "What is the status of pull request #1371" - info showing status not showing
6. "Show me what files changed in PR #1371 in the repository" ✅
7. "Show me issues related to 'memory leak' in the last 30 days" ✅
8. "List all files in the main branch of the repository" ✅
9. "Show me all files in the 'feature/new-ui' branch" ✅
10. "List all closed pull requests" ✅

### Pending Test Cases
1. "Comment on issue #477 with 'This is a helpful comment about the bug'"
2. "Create a pull request titled 'Fix performance bug' with description 'This fixes the memory leak issue', from branch 'fix/performance' to 'main'"
3. "Read the contents of README.md file"
4. "Create a new file 'docs/contributing.md' with content '# Contributing Guidelines\n\nPlease follow these guidelines when contributing to the project.'"
5. "Update the file 'src/index.js' with new content 'console.log(\"Hello World\");'"
6. "Delete the file 'temp/deprecated.js'"
7. "Read the contents of 'src/utils/helpers.js' from the 'develop' branch"

## Notion-Only Queries (New Functionality)

### Database Operations
1. "List all databases in my Notion workspace" ✅
2. "Find databases related to 'interview'" ✅
3. "Get details of the database with ID 1bb04458-70d1-80ce-a7c0-fa0c3ee81b39" ✅
4. "Get details of the database DP Interview Prep" ✅
5. "Show entries in the DP Interview Prep database"✅
6. "List all items in the Products database"✅
7. "Query the 'Tasks' database for open items assigned to me"
8. "Query the 'Products' database for items with priority 'Medium'" ✅
9. "Show me all entries in the database with ID abc123-def456-789"
10. "Get database contents for dp interview prep"✅ lowercase working
11. "List pages in the Projects database"
12. "Query Tasks database where Status is 'In Progress'"
13. "Show me items in Products database with priority High"
14. "Get all entries from the Team database"
15. "List database records for Bug Tracking"

### Page Operations
1. "Get details of the page with ID 'def456'"
2. "Update the page with ID 'def456' to set the status to 'Completed'"
3. "Create a new page in the 'Tasks' database with title 'Review documentation' and priority 'High'"
4. "Create a new page in Products database with name 'New Feature' and priority 'Medium'"
5. "Update page abc123 to change status to Done"
6. "Get information about the page titled 'Project Overview'"
7. "Create a page in Interview Prep database with title 'System Design Questions'"

### Search Operations
1. "Search for pages about 'Q3 planning'"
2. "Find all pages mentioning 'budget' in my workspace"
3. "Search for content related to 'team meeting notes'"
4. "Find pages containing 'API documentation'"
5. "Search Notion for 'sprint planning' content"
6. "Look for pages about 'database optimization'"

### Database Schema and Structure Queries
1. "What are the properties of the Tasks database?"
2. "Show me the structure of the Products database"
3. "Get database schema for Interview Prep"
4. "What fields does the Projects database have?"
5. "Show me the database properties for Bug Tracking"

## Mixed GitHub + Notion Queries (New Multi-Service Functionality)

### Synchronization Queries
1. "Find all open GitHub issues and create a Notion page for each in my 'Tasks' database"
2. "Search for Notion pages about 'bug fixes' and create GitHub issues for each"
3. "List all GitHub pull requests and create a summary page in my Notion 'Development' database"
4. "Get the details of GitHub issue #123 and create a Notion page with the same information"
5. "Find Notion pages tagged as 'urgent' and create GitHub issues for them in the 'bug-reports' repository"
6. "Sync GitHub issues labeled 'documentation' with Notion docs database"
7. "Create Notion tasks for all GitHub PRs waiting for review"

### Reporting Queries
1. "Create a weekly report that includes open GitHub issues and Notion tasks due this week"
2. "Generate a project status report combining GitHub PR status and Notion project tracking"
3. "Show me all GitHub issues assigned to me and Notion tasks in the 'In Progress' column"
4. "Compare GitHub milestone progress with Notion project status"
5. "Generate a summary of closed GitHub issues and completed Notion tasks from last month"

### Workflow Automation
1. "When I create a Notion page in the 'Ideas' database, automatically create a GitHub issue"
2. "When a GitHub issue is closed, update the corresponding Notion page status to 'Done'"
3. "Sync GitHub repository files with a Notion database of documentation"
4. "Create GitHub issues from Notion pages marked as 'Bug'"
5. "Update Notion project status when GitHub PR is merged"

## Enhanced Database Detail Queries (Testing Rich API Response)

### Comprehensive Database Information
1. "Get complete details of database DP Interview Prep including all entries"
2. "Show me everything about the Products database - schema and data"
3. "Get full details of Tasks database with sample entries"
4. "Display comprehensive information for Bug Tracking database"
5. "Show database structure and first 5 entries for Projects"
6. "Get detailed view of Team database including properties and pages"

### Filtered Database Queries
1. "Show entries in Products database where priority equals High"
2. "Get Tasks database items with status In Progress"
3. "Query Interview Prep database for items created this week"
4. "Show Bug Tracking entries assigned to me"
5. "Get Projects database items due before next month"
6. "Find Team database entries with role Developer"

### Sorting and Organization
1. "Get Tasks database sorted by priority descending"
2. "Show Products database ordered by creation date"
3. "List Interview Prep entries sorted by difficulty level"
4. "Get Bug Tracking items ordered by severity"
5. "Show Projects database sorted by due date ascending"

## Error Handling and Edge Cases

### Invalid Database References
1. "Get details of database NonExistentDB"
2. "Query database with ID invalid-uuid-format"
3. "Show entries in database that doesn't exist"

### Permission and Access Issues
1. "Access database I don't have permission to view"
2. "Query shared database with limited access"
3. "Get details of archived database"

### Empty or Minimal Data
1. "Show entries in empty database"
2. "Get details of database with no properties"
3. "Query database with no matching results"