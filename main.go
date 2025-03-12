package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const (
	testFileSuffix = "_test.go"
)

type QA struct {
	Question   string      `json:"question"`
	Answer     interface{} `json:"answer"` // Changed to interface{} to allow structured answers
	SourceFile string      `json:"source_file"`
	TestName   string      `json:"test_name"`
}

type StructuredAnswer struct {
	Purpose          string                   `json:"purpose"`
	Assertions       []map[string]interface{} `json:"assertions"`
	ExpectedBehavior string                   `json:"expected_behavior"`
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go_ingestion_agent <directory>")
	}
	testDir := os.Args[1]
	qaPairs, err := processGoTests(testDir)
	if err != nil {
		log.Fatalf("Error processing go tests: %v", err)
	}

	jsonData, err := json.Marshal(qaPairs)
	if err != nil {
		log.Fatalf("Error encoding JSON: %v", err)
	}
	fmt.Println(string(jsonData))
}

func processGoTests(dirPath string) ([]QA, error) {
	var qaPairs []QA
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("error walking path %s: %w", path, err)
		}
		if !strings.HasSuffix(info.Name(), testFileSuffix) {
			return nil
		}
		log.Printf("Processing test file: %s", path)
		content, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("error reading file %s: %w", path, err)
		}
		// Load the ast
		fset := token.NewFileSet()
		node, err := parser.ParseFile(fset, path, content, parser.ParseComments)
		if err != nil {
			return fmt.Errorf("error parsing file %s: %w", path, err)
		}

		//Find the test function declerations
		testsFound := false
		for _, decl := range node.Decls {
			if funcDecl, ok := decl.(*ast.FuncDecl); ok && strings.HasPrefix(funcDecl.Name.Name, "Test") {
				testsFound = true
				testName := strings.TrimSpace(funcDecl.Name.Name)

				// Get comment above function
				var doc strings.Builder
				if funcDecl.Doc != nil {
					for _, comment := range funcDecl.Doc.List {
						doc.WriteString(comment.Text)
					}
				}

				//Here we could check for assertions in the body. For now just return a structed answer placeholder.
				structuredAnswer := StructuredAnswer{
					Purpose:          "This is a placeholder. This test is testing something. In a full implementation we would analyse the code to see what is tested and asserted.",
					Assertions:       []map[string]interface{}{},
					ExpectedBehavior: "This is a placeholder. In a full implementation we would analyse the code to see what is expected.",
				}

				qaPairs = append(qaPairs, QA{
					Question:   fmt.Sprintf("What does the test function '%s' in file '%s' do? %s", testName, path, doc.String()),
					Answer:     structuredAnswer,
					SourceFile: path,
					TestName:   testName,
				})
			}
		}
		if !testsFound {
			log.Printf("No tests found in file: %s", path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error processing go tests in directory %s: %w", dirPath, err)
	}

	return qaPairs, nil
}
